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

# --- 1. Helper Function from training script ---
def smooth_scale(x):
    # This matches the 'smooth_scale' import in your modelv9.py
    return torch.sigmoid(x) 

# --- 2. All required classes from modelv9.py ---
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
        self.layers = nn.Sequential(
            MultiKernelResBlock(in_channels, base_channels, downsample=True),
            MultiKernelResBlock(base_channels, base_channels * 2, downsample=True),
            MultiKernelResBlock(base_channels * 2, base_channels * 4, downsample=True),
        )
        conv_dim = int(img_size[0] * img_size[1] * base_channels * 4 / (8 * 8))
        self.fc_mu = nn.Linear(conv_dim, latent_size)
        self.fc_logvar = nn.Linear(conv_dim, latent_size)

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
    def forward(self, x): return self.layers(x)

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
        self.encoder = ResEncoder((image_size[1], image_size[2]), self.C, hidden_dimension, latent_size)
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
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu_q, logvar_q = self.encoder(x)
        z = self.reparameterize(mu_q, logvar_q)
        return self.decoder(z), mu_q, logvar_q, self.ctr_head(z), z

# --- 3. DUMMY MODULE SETUP ---
# This is the "brain transplant" that makes torch.load work
m = types.ModuleType("src")
m.modelv9 = types.ModuleType("modelv9")
# Register ALL classes that were in modelv9.py
m.modelv9.VAE = VAE 
m.modelv9.ResEncoder = ResEncoder
m.modelv9.ResDecoder = ResDecoder
m.modelv9.MultiKernelResBlock = MultiKernelResBlock
m.modelv9.ResUpBlock = ResUpBlock
m.modelv9.GroupResUpBlock = GroupResUpBlock
m.modelv9.ChannelWiseBlock = ChannelWiseBlock

sys.modules["src"] = m
sys.modules["src.modelv9"] = m.modelv9

# --- 4. Streamlit Load Function ---
@st.cache_resource
def load_model():
    model_path = os.path.join("knowledge-base", "vae_model.pt")
    if not os.path.exists(model_path):
        st.error(f"File not found: {model_path}")
        return None
    try:
        # Load the full object now that all components are registered
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Streamlit app
st.title("VAE Image Reconstruction App")

# Instructions
st.write("Upload a 64x64 image (or it will be resized) to reconstruct it using the trained VAE model. The app will also predict control parameters (regression output).")

# Load model
model = load_model()
if model is None:
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and process image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Transform: resize to 64x64, to tensor, normalize (assuming [0,1] range as in training)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    x = transform(image).unsqueeze(0)  # Add batch dim
    
    # Run inference
    with torch.no_grad():
        recon, _, _, ctr_pred, _ = model(x)
    
    # Display reconstructed image
    recon_img = recon.squeeze(0)  # Remove batch dim
    recon_pil = transforms.ToPILImage()(recon_img)
    st.image(recon_pil, caption="Reconstructed Image", use_column_width=True)
    
    # Display regression output (control predictions)
    st.subheader("Predicted Control Parameters")
    ctr_array = ctr_pred.squeeze(0).numpy()  # Assuming batch size 1
    st.write(ctr_array)  # Or format as a table/list if needed
