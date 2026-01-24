import os, io
import torch, cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

from src.dataloader import smooth_scale, inv_smooth_scale
from src.modelv11 import mdn_point_and_confidence


@st.cache_resource
def load_model(device="cpu"):
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
        device = torch.device(device)

        # Try to load the state dict
        try:
            vae = torch.load(combined_data, map_location=device, weights_only=False)
            vae.eval()
            st.success("✅ Model loaded successfully!")
            return vae

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    except Exception as e:
        st.error(f"Error reassembling model: {str(e)}")
        return None

# ==========================================================
# HELPER FUNCTIONS FOR IMAGE HANDLING
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
            image_extensions = ['.npy', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_files = []

            for ext in image_extensions:
                image_files.extend(Path(test_folder).glob(f"*{ext}"))
                image_files.extend(Path(test_folder).glob(f"*{ext.upper()}"))

            if image_files:
                return test_folder, sorted(image_files)

    return None, []

# ==========================================================
# HELPER FUNCTIONS FOR MODEL INFERENCE
# ==========================================================
def load_image_from_path(image_path):
    """Load image from file path"""
    try:
        if image_path.name.endswith(".npy"):
            return np.load(image_path)
        else:
            return np.array(Image.open(image_path).convert("RGB")) / 255.
    except Exception as e:
        st.error(f"Error loading image {image_path}: {str(e)}")
        return None

def prep_tensor_from_image(image: np.ndarray, image_size: tuple):
    """image: (H,W,3) float in [0,1] or npy compatible; returns (1,3,H,W) torch tensor after smooth_scale"""
    arr = cv2.resize(image, image_size)
    tensor_t = torch.from_numpy(arr).float().permute(2, 0, 1)
    tensor_t = smooth_scale(tensor_t)
    return tensor_t[None]

def process_image(image, model, image_size, var_scale):
    """Process image through the model"""
    original_shape = image.shape
    tensor_t = prep_tensor_from_image(image, image_size)

    with torch.no_grad():
        recon, _, _, (pi_s, mu_s, log_sigma_s), _ = model(tensor_t)

    # Ensure reconstruction is in valid range
    recon_img = inv_smooth_scale(recon)     # post processing
    recon_img = recon_img.detach().cpu().numpy()[0].transpose(1, 2, 0)
    recon_img = cv2.resize(recon_img, (original_shape[1], original_shape[0]))

    # Get control parameters
    theta_hat_s, conf_param_s, conf_global_s, modes_s = mdn_point_and_confidence(
        pi_s, mu_s, log_sigma_s, var_scale=var_scale, topk=3
    )
    y_pred_s = theta_hat_s.detach().cpu().numpy()[0]
    conf_s = conf_param_s.detach().cpu().numpy()[0]
    conf_global_s = conf_global_s.detach().cpu().numpy()[0]

    return recon_img, y_pred_s, conf_s, conf_global_s

@torch.no_grad()
def inference(model, z, var_scale: float = 1.0, topk: int = 3):
    """Inference from z"""

    recon = model.decoder(z)

    pi, mu, log_sigma = model.mdn_head(z)
    theta_hat, conf_param, conf_global, modes = mdn_point_and_confidence(
        pi, mu, log_sigma, var_scale=var_scale, topk=topk
    )

    return recon, (theta_hat, conf_param, conf_global, modes)