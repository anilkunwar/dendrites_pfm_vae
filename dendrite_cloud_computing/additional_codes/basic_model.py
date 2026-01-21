import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import sys
import types
import math
import io

# ==========================================================
# 1. ARCHITECTURE DEFINITIONS (Fixed Version)
# ==========================================================
def smooth_scale(x):
    return torch.sigmoid(x)

class MultiKernelResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=False)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, stride, 3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = (self.conv3(x) + self.conv5(x) + self.conv7(x)) / 3.0
        out = self.bn(out)
        out = out + self.shortcut(x)
        return self.relu(out)

class ResEncoder(nn.Module):
    def __init__(self, img_size, in_channels, base_channels, latent_size):
        super().__init__()
        self.img_size = img_size
        self.base_channels = base_channels
        
        self.layers = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True),
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),
        )
        
        # Calculate the flattened dimension dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_size[0], img_size[1])
            dummy_output = self.layers(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
        
        self.fc_mu = nn.Linear(flattened_size, latent_size)
        self.fc_logvar = nn.Linear(flattened_size, latent_size)
    
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return self.fc_mu(x), self.fc_logvar(x)

class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.deconv1(x)))
        out = self.bn2(self.conv(out))
        return self.relu(out + self.shortcut(x))

class GroupResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.short = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.deconv(x)))
        out = self.bn2(self.conv(out))
        return self.relu(out + self.short(x))

class ChannelWiseBlock(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.layers(x)

class ResDecoder(nn.Module):
    def __init__(self, out_channels, base_channels, latent_size, H, W):
        super().__init__()
        self.H, self.W = H, W
        init_h, init_w = H // 8, W // 8
        self.fc = nn.Linear(latent_size, base_channels * 4 * init_h * init_w)
        
        self.shared_up = nn.Sequential(
            ResUpBlock(base_channels * 4, base_channels * 2),
            ResUpBlock(base_channels * 2, base_channels),
        )
        
        ch_per_group = math.ceil(base_channels / out_channels)
        self.group_channels = ch_per_group * out_channels
        
        self.align_proj = nn.Conv2d(base_channels, self.group_channels, kernel_size=1, bias=False)
        self.align_bn = nn.BatchNorm2d(self.group_channels)
        
        self.group_up = GroupResUpBlock(self.group_channels, self.group_channels, groups=out_channels)
        
        self.refine = nn.Sequential(
            ChannelWiseBlock(self.group_channels, groups=out_channels),
            ChannelWiseBlock(self.group_channels, groups=out_channels),
        )
        
        self.head = nn.Conv2d(self.group_channels, out_channels, 1, groups=out_channels, bias=True)
    
    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, self.H // 8, self.W // 8)
        x = self.shared_up(x)
        x = self.align_bn(self.align_proj(x))
        x = self.group_up(x)
        x = self.refine(x)
        return smooth_scale(self.head(x))

class VAE(nn.Module):
    def __init__(self, image_size=(3, 128, 128), latent_size=64, hidden_dimension=64, num_params=15, ctr_head_hidden=256):
        super().__init__()
        self.C, self.H, self.W = image_size
        self.latent_size = latent_size
        
        self.encoder = ResEncoder((self.H, self.W), self.C, hidden_dimension, latent_size)
        self.decoder = ResDecoder(self.C, hidden_dimension, latent_size, self.H, self.W)
        
        self.ctr_head = nn.Sequential(
            nn.Linear(latent_size, ctr_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(ctr_head_hidden, ctr_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(ctr_head_hidden, num_params),
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu_q, logvar_q = self.encoder(x)
        z = self.reparameterize(mu_q, logvar_q)
        recon = self.decoder(z)
        ctr_pred = self.ctr_head(z)
        return recon, mu_q, logvar_q, ctr_pred, z

# ==========================================================
# 2. DUMMY MODULE SETUP
# ==========================================================
m = types.ModuleType("src")
m.modelv9 = types.ModuleType("modelv9")
m.modelv9.VAE = VAE
m.modelv9.ResEncoder = ResEncoder
m.modelv9.ResDecoder = ResDecoder
m.modelv9.MultiKernelResBlock = MultiKernelResBlock
m.modelv9.ResUpBlock = ResUpBlock
m.modelv9.GroupResUpBlock = GroupResUpBlock
m.modelv9.ChannelWiseBlock = ChannelWiseBlock
sys.modules["src"] = m
sys.modules["src.modelv9"] = m.modelv9

# ==========================================================
# 3. STREAMLIT LOADING
# ==========================================================
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible locations
    possible_folders = ["knowledge_base", "knowledge-base", ".", "models"]
    folder_found = None
    
    for f in possible_folders:
        test_path = os.path.join(current_dir, f)
        if os.path.exists(test_path) and os.path.isdir(test_path):
            # Check if part1 exists
            part1_path = os.path.join(test_path, "vae_model.pt.part1")
            if os.path.exists(part1_path):
                folder_found = test_path
                break
    
    if folder_found is None:
        st.error("❌ Could not find the model parts.")
        st.info("Please ensure the model files are in one of these folders: " + ", ".join(possible_folders))
        return None
    
    base_name = "vae_model.pt"
    num_parts = 4
    parts = [os.path.join(folder_found, f"{base_name}.part{i}") for i in range(1, num_parts + 1)]
    
    try:
        combined_data = io.BytesIO()
        with st.spinner(f"Merging model parts from {os.path.basename(folder_found)}..."):
            for p in parts:
                if not os.path.exists(p):
                    st.error(f"Missing: {p}")
                    return None
                with open(p, 'rb') as f:
                    combined_data.write(f.read())
        
        combined_data.seek(0)
        
        # Load the model
        device = torch.device('cpu')
        
        # First, create a dummy model to get the architecture
        dummy_model = VAE(image_size=(3, 128, 128), latent_size=64, hidden_dimension=64)
        
        # Try to load the state dict
        try:
            checkpoint = torch.load(combined_data, map_location=device, weights_only=False)
            
            # Check if it's a state dict or full model
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'module.' prefix if present (for DataParallel models)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                dummy_model.load_state_dict(state_dict, strict=False)
            elif isinstance(checkpoint, dict):
                # Assume it's a state dict
                checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
                dummy_model.load_state_dict(checkpoint, strict=False)
            else:
                # It's a full model
                dummy_model = checkpoint
            
            dummy_model.eval()
            st.success("✅ Model loaded successfully!")
            return dummy_model
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error reassembling model: {str(e)}")
        return None

# ==========================================================
# 4. STREAMLIT UI & INFERENCE
# ==========================================================
st.title("VAE Image Reconstruction App")
st.write("Upload an image to reconstruct it and predict control parameters.")

model = load_model()

if model is None:
    st.stop()

# Get model's expected input size
if hasattr(model, 'H') and hasattr(model, 'W'):
    expected_h, expected_w = model.H, model.W
else:
    # Default to 128x128 if not specified
    expected_h, expected_w = 128, 128

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Transform to match model's expected input size
        transform = transforms.Compose([
            transforms.Resize((expected_h, expected_w)),
            transforms.ToTensor(),
        ])
        
        x = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            recon, _, _, ctr_pred, _ = model(x)
        
        # Ensure reconstruction is in valid range
        recon_img = torch.clamp(recon.squeeze(0), 0, 1)
        recon_pil = transforms.ToPILImage()(recon_img)
        
        with col2:
            st.image(recon_pil, caption="Reconstructed Image", use_column_width=True)
        
        st.subheader("Predicted Control Parameters")
        ctr_array = ctr_pred.squeeze(0).numpy()
        
        # Display as a table with parameter indices
        import pandas as pd
        df = pd.DataFrame({
            "Parameter Index": list(range(len(ctr_array))),
            "Value": ctr_array
        })
        st.table(df)
        
        # Show as bar chart using Streamlit's built-in chart
        st.subheader("Parameter Visualization")
        st.bar_chart(ctr_array)
        
        # Optionally show statistics
        st.subheader("Parameter Statistics")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Mean", f"{np.mean(ctr_array):.4f}")
        with col_stats2:
            st.metric("Std Dev", f"{np.std(ctr_array):.4f}")
        with col_stats3:
            st.metric("Range", f"{np.max(ctr_array)-np.min(ctr_array):.4f}")
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try another image or check the model compatibility.")

# Add some helpful information
st.sidebar.header("About")
st.sidebar.info("""
This app uses a Variational Autoencoder (VAE) to:
1. Reconstruct uploaded images
2. Predict control parameters from image features

The model expects images of size 128x128 pixels.
""")

# Add download option for reconstructed image
if 'recon_pil' in locals():
    st.sidebar.header("Download")
    buf = io.BytesIO()
    recon_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.sidebar.download_button(
        label="Download Reconstructed Image",
        data=byte_im,
        file_name="reconstructed.png",
        mime="image/png"
    )
