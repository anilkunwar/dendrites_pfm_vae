import os
import torch
import cv2
import streamlit as st
import numpy as np
import io
import pandas as pd
from pathlib import Path
from PIL import Image
from matplotlib import colors, cm

from src.dataloader import inv_scale_params, smooth_scale, PARAM_RANGES
from src.modelv11 import mdn_point_and_confidence

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
            image_extensions = ['.npy', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
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
        if image_path.endswith(".npy"):
            return np.load(image_path)
        else:
            return Image.open(image_path).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image {image_path}: {str(e)}")
        return None


def process_image(image, model, image_size):
    """Process image through the model"""

    arr = cv2.resize(image, image_size)
    tensor_t = torch.from_numpy(arr).float().permute(2, 0, 1)
    tensor_t = smooth_scale(tensor_t)

    with torch.no_grad():
        recon, _, _, (pi_s, mu_s, log_sigma_s), _ = model(tensor_t[None])

    # Ensure reconstruction is in valid range
    recon_img = recon.detach().cpu().numpy()[0]

    # Get control parameters
    theta_hat_s, conf_param_s, conf_global_s, modes_s = mdn_point_and_confidence(
        pi_s, mu_s, log_sigma_s, var_scale=1, topk=3
    )
    y_pred_s = theta_hat_s.detach().cpu().numpy()[0]
    conf_s = conf_param_s.detach().cpu().numpy()[0]
    conf_global_s = conf_global_s.detach().cpu().numpy()[0]

    return recon_img, y_pred_s


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
    expected_size = (model.H, model.W)
else:
    expected_size = (48, 48)

param_names = ["t"]
param_names += list(PARAM_RANGES.keys())

# Main interface with tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📂 Select from Test Images", "📊 Batch Analysis"])

def show_coolwarm(gray_image, caption):
    norm = colors.Normalize(vmin=gray_image.min(), vmax=gray_image.max())
    colored_img = cm.coolwarm(norm(gray_image))  # shape: (H, W, 4)
    st.image(colored_img, caption=caption, use_column_width=True)

with tab1:
    st.header("Upload Your Own Image (Note your images should be PFM results (eta, c, potential) with valid data range")
    uploaded_file = st.file_uploader("Choose an image file...", type=[".npy", "jpg", "png", "jpeg", "bmp", "tiff"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".npy"):
                bytes_data = uploaded_file.getvalue()
                buffer = io.BytesIO(bytes_data)
                image = np.load(buffer)
            else:
                image = np.array(Image.open(uploaded_file).convert("RGB"))

            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image (Only 1st channel: order parameter)")
                show_coolwarm(image[..., 0], caption=f"Uploaded: {uploaded_file.name}")
                st.caption(f"Size: {image.shape[0]}×{image.shape[1]}")

            # Process image
            recon_image, ctr_array = process_image(image, model, expected_size)

            # Display reconstruction
            with col2:
                st.subheader(f"Reconstructed Image (Only 1st channel: order parameter) {recon_image.shape}")
                show_coolwarm(recon_image[..., 0], caption="VAE Reconstruction")
                st.caption(f"Resized to: {expected_size}")

            # Display control parameters
            st.subheader("📈 Predicted Control Parameters")

            # Create parameter table
            param_df = pd.DataFrame({
                "Parameter": param_names,
                "Predict Value (Normalized)": ctr_array,
                "Predict Value (Denormalized)": inv_scale_params(ctr_array),
            })

            col_table, col_chart = st.columns([1, 2])

            with col_table:
                st.dataframe(param_df.style.format({"Predict Value (Normalized)": "{:.4f}", "Predict Value (Denormalized)": "{:.9f}"}))

            with col_chart:
                st.bar_chart(param_df.set_index("Parameter")["Predict Value (Normalized)"])

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
            Image.fromarray(recon_image).convert("RGB").save(buf, format="PNG")
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
                recon_pil, ctr_array = process_image(image, model, expected_size)

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
                    compare_df.index = param_names

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
                        recon_pil, ctr_array = process_image(image, model, expected_size)

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
