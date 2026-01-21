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
import pandas as pd
from pathlib import Path

# ==========================================================
# 1. ARCHITECTURE DEFINITIONS
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
# 4. HELPER FUNCTIONS FOR IMAGE HANDLING
# ==========================================================
def get_test_images():
    """Scan for test_input folder and return available images"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible folder names
    possible_folders = ["test_input", "test_images", "images", "test"]
    
    for folder_name in possible_folders:
        test_folder = os.path.join(current_dir, folder_name)
        if os.path.exists(test_folder) and os.path.isdir(test_folder):
            # Find all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(test_folder).glob(f"*{ext}"))
                image_files.extend(Path(test_folder).glob(f"*{ext.upper()}"))
            
            if image_files:
                return test_folder, sorted(image_files)
    
    return None, []

def load_image_from_path(image_path):
    """Load image from file path"""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Error loading image {image_path}: {str(e)}")
        return None

def process_image(image, model, expected_h=128, expected_w=128):
    """Process image through the model"""
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
    
    # Get control parameters
    ctr_array = ctr_pred.squeeze(0).numpy()
    
    return recon_pil, ctr_array

# ==========================================================
# 5. STREAMLIT UI & INFERENCE
# ==========================================================
st.set_page_config(layout="wide", page_title="VAE Image Reconstruction")

st.title("🎨 VAE Image Reconstruction & Analysis")
st.markdown("Upload an image or select from test images to reconstruct it and analyze predicted control parameters.")

# Sidebar for model info and controls
with st.sidebar:
    st.header("⚙️ Controls & Information")
    
    st.markdown("### Model Information")
    st.info("""
    This VAE model:
    - Reconstructs 128×128 RGB images
    - Predicts 15 control parameters
    - Uses a multi-kernel residual architecture
    """)
    
    # Check for test images
    test_folder, test_images = get_test_images()
    
    if test_images:
        st.markdown("### Available Test Images")
        test_image_names = [img.name for img in test_images]
        st.info(f"Found {len(test_images)} images in '{os.path.basename(test_folder)}' folder")
    else:
        st.warning("No test images found. Create a 'test_input' folder with images.")

# Load model
model = load_model()

if model is None:
    st.stop()

# Get model's expected input size
if hasattr(model, 'H') and hasattr(model, 'W'):
    expected_h, expected_w = model.H, model.W
else:
    expected_h, expected_w = 128, 128

# Main interface with tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📂 Select from Test Images", "📊 Batch Analysis"])

with tab1:
    st.header("Upload Your Own Image")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg", "bmp", "tiff"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                st.caption(f"Size: {image.size[0]}×{image.size[1]}, Mode: {image.mode}")
            
            # Process image
            recon_pil, ctr_array = process_image(image, model, expected_h, expected_w)
            
            # Display reconstruction
            with col2:
                st.subheader("Reconstructed Image")
                st.image(recon_pil, caption="VAE Reconstruction", use_column_width=True)
                st.caption(f"Resized to: {expected_w}×{expected_h}")
            
            # Display control parameters
            st.subheader("📈 Predicted Control Parameters")
            
            # Create parameter table
            param_df = pd.DataFrame({
                "Parameter": [f"P{i:02d}" for i in range(len(ctr_array))],
                "Value": ctr_array,
                "Normalized": (ctr_array - ctr_array.min()) / (ctr_array.max() - ctr_array.min() + 1e-8)
            })
            
            col_table, col_chart = st.columns([1, 2])
            
            with col_table:
                st.dataframe(param_df.style.format({"Value": "{:.4f}", "Normalized": "{:.3f}"}))
            
            with col_chart:
                st.bar_chart(param_df.set_index("Parameter")["Value"])
            
            # Parameter statistics
            st.subheader("📊 Parameter Statistics")
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                st.metric("Mean", f"{np.mean(ctr_array):.4f}")
            with stats_col2:
                st.metric("Std Dev", f"{np.std(ctr_array):.4f}")
            with stats_col3:
                st.metric("Min", f"{np.min(ctr_array):.4f}")
            with stats_col4:
                st.metric("Max", f"{np.max(ctr_array):.4f}")
            
            # Download button
            st.markdown("---")
            buf = io.BytesIO()
            recon_pil.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Reconstructed Image",
                data=buf.getvalue(),
                file_name="reconstructed_image.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with tab2:
    st.header("Select from Test Images")
    
    if test_images:
        # Create image selector
        image_names = [img.name for img in test_images]
        selected_image_name = st.selectbox("Choose a test image:", image_names)
        
        if selected_image_name:
            # Find the selected image path
            selected_idx = image_names.index(selected_image_name)
            selected_path = test_images[selected_idx]
            
            # Load and display the image
            image = load_image_from_path(selected_path)
            
            if image:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Test Image")
                    st.image(image, caption=f"Selected: {selected_image_name}", use_column_width=True)
                    st.caption(f"Path: {selected_path}")
                
                # Process image
                recon_pil, ctr_array = process_image(image, model, expected_h, expected_w)
                
                with col2:
                    st.subheader("Reconstructed Image")
                    st.image(recon_pil, caption="VAE Reconstruction", use_column_width=True)
                
                # Display control parameters
                st.subheader("📈 Predicted Control Parameters")
                
                param_df = pd.DataFrame({
                    "Parameter": [f"P{i:02d}" for i in range(len(ctr_array))],
                    "Value": ctr_array
                })
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.bar_chart(param_df.set_index("Parameter")["Value"])
                
                with col_chart2:
                    # Create a line chart for parameter trends
                    st.line_chart(param_df.set_index("Parameter")["Value"])
                
                # Show parameter table
                st.dataframe(param_df.style.format({"Value": "{:.4f}"}))
                
                # Quick comparison if multiple images have been processed
                if 'previous_params' not in st.session_state:
                    st.session_state.previous_params = {}
                
                if st.button("💾 Save these parameters for comparison"):
                    st.session_state.previous_params[selected_image_name] = ctr_array
                    st.success(f"Saved parameters for {selected_image_name}")
                
                # Show saved parameters for comparison
                if st.session_state.previous_params:
                    st.subheader("📋 Saved Parameter Comparisons")
                    
                    # Create comparison DataFrame
                    compare_data = {}
                    for img_name, params in st.session_state.previous_params.items():
                        compare_data[img_name] = params
                    
                    compare_df = pd.DataFrame(compare_data)
                    compare_df.index = [f"P{i:02d}" for i in range(len(ctr_array))]
                    
                    st.dataframe(compare_df.style.format("{:.4f}"))
                    
                    if len(st.session_state.previous_params) > 1:
                        st.line_chart(compare_df.T)
    else:
        st.warning("No test images found. Please create a 'test_input' folder with images.")
        st.info("""
        To use this feature:
        1. Create a folder named 'test_input' in the same directory as this script
        2. Add some images (jpg, png, etc.) to the folder
        3. Refresh the app
        """)

with tab3:
    st.header("Batch Image Analysis")
    
    if test_images:
        st.info(f"Found {len(test_images)} images in test folder. Select which ones to analyze.")
        
        # Multi-select for batch processing
        selected_images = st.multiselect(
            "Select images for batch analysis:",
            options=[img.name for img in test_images],
            default=[img.name for img in test_images[:3]] if len(test_images) >= 3 else []
        )
        
        if selected_images and st.button("🚀 Run Batch Analysis"):
            with st.spinner("Processing images..."):
                results = []
                progress_bar = st.progress(0)
                
                for idx, img_name in enumerate(selected_images):
                    # Find and load image
                    img_path = test_images[[img.name for img in test_images].index(img_name)]
                    image = load_image_from_path(img_path)
                    
                    if image:
                        # Process image
                        recon_pil, ctr_array = process_image(image, model, expected_h, expected_w)
                        
                        # Store results
                        result = {
                            "Image": img_name,
                            "Mean": np.mean(ctr_array),
                            "Std": np.std(ctr_array),
                            "Min": np.min(ctr_array),
                            "Max": np.max(ctr_array),
                            "Params": ctr_array
                        }
                        results.append(result)
                    
                    progress_bar.progress((idx + 1) / len(selected_images))
                
                if results:
                    st.success(f"✅ Processed {len(results)} images")
                    
                    # Display summary statistics
                    st.subheader("📊 Batch Summary Statistics")
                    
                    summary_df = pd.DataFrame([{
                        "Image": r["Image"],
                        "Mean": r["Mean"],
                        "Std": r["Std"],
                        "Range": r["Max"] - r["Min"]
                    } for r in results])
                    
                    st.dataframe(summary_df.style.format({"Mean": "{:.4f}", "Std": "{:.4f}", "Range": "{:.4f}"}))
                    
                    # Create parameter matrix
                    st.subheader("🔢 Full Parameter Matrix")
                    
                    param_matrix = pd.DataFrame([r["Params"] for r in results], 
                                               index=[r["Image"] for r in results],
                                               columns=[f"P{i:02d}" for i in range(len(ctr_array))])
                    
                    st.dataframe(param_matrix.style.format("{:.4f}"))
                    
                    # Heatmap visualization
                    st.subheader("🔥 Parameter Heatmap")
                    
                    # Normalize for visualization
                    param_matrix_normalized = (param_matrix - param_matrix.min().min()) / \
                                            (param_matrix.max().max() - param_matrix.min().min())
                    
                    # Display as a styled table (heatmap approximation)
                    st.dataframe(param_matrix_normalized.style.format("{:.2f}").background_gradient(cmap="viridis"))
                    
                    # Download results as CSV
                    csv = param_matrix.to_csv()
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="batch_analysis_results.csv",
                        mime="text/csv"
                    )
    else:
        st.warning("No test images found for batch analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>VAE Image Reconstruction App • Built with PyTorch & Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# 6. APP CONFIGURATION NOTES
# ==========================================================
"""
APP CONFIGURATION:

Folder Structure Expected:
├── app.py (this file)
├── knowledge_base/ (or knowledge-base/)
│   ├── vae_model.pt.part1
│   ├── vae_model.pt.part2
│   ├── vae_model.pt.part3
│   └── vae_model.pt.part4
├── test_input/ (optional)
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── (other project files)



Features:
- Upload custom images
- Select from test images in test_input folder
- Batch process multiple images
- View parameter visualizations
- Download results
"""
