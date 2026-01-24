import os
import torch
import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd
from pathlib import Path
from PIL import Image
from matplotlib import colors, cm

from src.evaluate_metrics import generate_analysis_figure
from src.dataloader import inv_scale_params, smooth_scale, inv_smooth_scale, PARAM_RANGES
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
        if image_path.name.endswith(".npy"):
            return np.load(image_path)
        else:
            return np.array(Image.open(image_path).convert("RGB")) / 255.
    except Exception as e:
        st.error(f"Error loading image {image_path}: {str(e)}")
        return None

def _prep_tensor_from_image(image: np.ndarray, image_size: tuple):
    """image: (H,W,3) float in [0,1] or npy compatible; returns (1,3,H,W) torch tensor after smooth_scale"""
    arr = cv2.resize(image, image_size)
    tensor_t = torch.from_numpy(arr).float().permute(2, 0, 1)
    tensor_t = smooth_scale(tensor_t)
    return tensor_t[None]

def process_image(image, model, image_size):
    """Process image through the model"""
    original_shape = image.shape
    tensor_t = _prep_tensor_from_image(image, image_size)

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

    var_scale = st.slider(
        "Var Scale (How much uncertainty is allowed)",
        0.01, 1.0, 0.1, 0.01,
        key="var_scale"
    )

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload Image", "📂 Select from Test Images", "📊 Batch Analysis", "🧪 Dendrite Intensity Score", "🔍 Heuristic Latent Space Exploration"])

def show_coolwarm(gray_image, caption):
    norm = colors.Normalize(vmin=gray_image.min(), vmax=gray_image.max())
    colored_img = cm.coolwarm(norm(gray_image))  # shape: (H, W, 4)
    st.image(colored_img, caption=caption, use_column_width=True)

def analyze_image(image, image_name:str):
    # Display original image (without preprocessing)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Original Image (Only 1st channel)")
        show_coolwarm(image[..., 0], caption=f"Selected: {image_name}")
        st.caption(
            f"Size: {image.shape[0]}×{image.shape[1]}, Max value: {np.max(image):.2f}, Min value: {np.min(image):.2f}")

    # Process image
    recon_image, ctr_array, conf_s, conf_global_s = process_image(image, model, expected_size)

    # Display reconstruction
    with col2:
        st.subheader(f"Reconstructed Image (Only 1st channel)")
        show_coolwarm(recon_image[..., 0], caption="VAE Reconstruction")
        st.caption(
            f"Resized from: {expected_size}, Max value: {np.max(recon_image[..., 0]):.2f}, Min value: {np.min(recon_image[..., 0]):.2f}")

    # Display control parameters
    st.subheader("📈 Predicted Control Parameters")

    # Create parameter table
    param_df = pd.DataFrame({
        "Parameter": param_names,
        "Predict Value (Normalized)": ctr_array,
        "Predict Value (Denormalized)": inv_scale_params(ctr_array),
        f"Confidence under var={var_scale}": conf_s
    })

    st.dataframe(
        param_df.style.format(
            {"Predict Value (Normalized)": "{:.4f}",
             "Predict Value (Denormalized)": "{:.9f}",
             f"Confidence under var={var_scale}": "{:.2f}"
             }))
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

    return recon_image, ctr_array

with tab1:
    st.header("Upload Your Own Image (Note your images should be PFM results (eta, c, potential) with valid data range")
    uploaded_file = st.file_uploader("Choose an image file...", type=["npy", "jpg", "png", "jpeg", "bmp", "tiff"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".npy"):
                bytes_data = uploaded_file.getvalue()
                buffer = io.BytesIO(bytes_data)
                image = np.load(buffer)
            else:
                image = np.array(Image.open(uploaded_file).convert("RGB")) / 255.   # convert to float

            recon_image, ctr_array = analyze_image(image, uploaded_file.name)

            # Download button
            st.markdown("---")
            buf = io.BytesIO()
            Image.fromarray((recon_image * 255).clip(0, 255).astype(np.uint8)).convert("RGB").save(buf, format="PNG")
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

            if image is not None:

                recon_image, ctr_array = analyze_image(image, selected_image_name)

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

                    if image is not None:
                        # Process image
                        recon_img, ctr_array, conf_s, conf_global_s = process_image(image, model, expected_size)

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
                                                columns=param_names)

                    st.dataframe(param_matrix.style.format("{:.4f}"))

                    # Heatmap visualization
                    st.subheader("🔥 Parameter Heatmap")

                    # Display as a styled table (heatmap approximation)
                    st.dataframe(param_matrix.style.format("{:.2f}").background_gradient(cmap="viridis"))

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

with tab4:
    st.header("Dendrite Intensity Analysis")

    # ========== 1) Session state for tab4 ==========
    session_item_id = 0
    st.session_state.tab4_items = []

    def _tab4_make_id(prefix: str, name: str) -> str:
        global session_item_id
        session_item_id += 1
        return f"{prefix}:{name}:{session_item_id}"

    def tab4_add_item(img: np.ndarray, name: str, source: str):
        result_img, _, scores = generate_analysis_figure(img[..., 0])
        st.session_state.tab4_items.append({
            "id": _tab4_make_id(source, name),
            "name": name,
            "source": source,
            "orig": img,
            "result": result_img,
            "score": scores["empirical_score"],
        })

    # ========== 2) Two-column UI ==========
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.subheader("📤 Upload Images")
        up_files = st.file_uploader(
            "Choose one or more image files...",
            type=["npy", "jpg", "png", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
            key="tab4_uploader",
        )
        if len(up_files) != len(st.session_state.tab4_items):
            for uf in up_files:
                if uf.name.endswith(".npy"):
                    buf = io.BytesIO(uf.getvalue())
                    img = np.load(buf)
                else:
                    img = np.array(Image.open(uf).convert("RGB")) / 255.0
                tab4_add_item(img, uf.name, source="upload")

        st.caption("Tip: you can upload multiple times. Newly uploaded images will be added to the list below.")

    with right_col:
        st.subheader("Select from Test Images")

        if test_images:
            test_names = [p.name for p in test_images]
            selected_dendrite_images = st.multiselect(
                "Select images for analysis:",
                options=[img.name for img in test_images],
                default=[]
            )

            if selected_dendrite_images:
                name_to_path = {p.name: p for p in test_images}
                for nm in selected_dendrite_images:
                    try:
                        img = load_image_from_path(name_to_path[nm])
                        tab4_add_item(img, nm, source="test")
                    except Exception as e:
                        st.error(f"Error loading image {nm}: {e}")
        else:
            st.warning("No test images found for analysis.")

    st.markdown("---")

    # ========== 3) Gallery: show + delete ==========
    st.subheader("🖼️ Analysis Statistics")

    if not st.session_state.tab4_items:
        st.info("No images added yet.")
    else:
        st.metric("Number of images", len(st.session_state.tab4_items))
        st.markdown("")

        with st.spinner("Processing images..."):
            progress_bar = st.progress(0)
            for idx, item in enumerate(st.session_state.tab4_items):
                container = st.container(border=True)
                with container:
                    header_cols = st.columns([1, 1])
                    with header_cols[0]:
                        st.markdown(f"**{item['name']}**  · from：`{item['source']}`")
                    with header_cols[1]:
                        st.metric("Score", f"{item['score']:.4f}")

                    st.pyplot(item["result"])
                progress_bar.progress((idx + 1) / len(st.session_state.tab4_items))
        st.success(f"✅ Processed {len(st.session_state.tab4_items)} images")

def _encode_image_get_z_and_recon(image: np.ndarray):
    """Run VAE once: get recon (RGB float, resized back to original), control params, and latent z (1D)."""
    original_shape = image.shape
    x = _prep_tensor_from_image(image, expected_size)  # (1,3,H,W)

    with torch.no_grad():
        # model forward returns: recon, mu_q, logvar_q, (pi_s, mu_s, log_sigma_s), z
        recon, _, _, (pi_s, mu_s, log_sigma_s), z = model(x)

    # recon to RGB float [0,1] and resize back to original
    recon_img = inv_smooth_scale(recon).detach().cpu().numpy()[0].transpose(1, 2, 0)
    recon_img = cv2.resize(recon_img, (original_shape[1], original_shape[0]))

    # control params for the sampled z
    theta_hat_s, conf_param_s, conf_global_s, modes_s = mdn_point_and_confidence(
        pi_s, mu_s, log_sigma_s, var_scale=var_scale, topk=3
    )
    y_pred_s = theta_hat_s.detach().cpu().numpy()[0]

    z_np = z.detach().cpu().numpy()
    # robust squeeze: could be (1,D) or (D,)
    z_np = np.squeeze(z_np)
    if z_np.ndim != 1:
        z_np = z_np.reshape(-1)

    return recon_img, y_pred_s, z_np

def _inference_from_z(z_1d: np.ndarray):
    """Decode from latent z; return recon_img (RGB float), y_pred_s (params)."""
    z_tensor = torch.from_numpy(z_1d.astype(np.float32))[None]  # (1,D)
    with torch.no_grad():
        # model.inference returns: recon, (theta_hat_s, conf_param_s, conf_global_s, modes_s)
        recon_cand, (theta_hat_s_cand, conf_param_s_cand, conf_global_s_cand, modes_s_cand) = \
            model.inference(z_tensor, var_scale=var_scale)

    recon_img = inv_smooth_scale(recon_cand).detach().cpu().numpy()[0].transpose(1, 2, 0)
    # NOTE: inference recon is in model's native resolution; we keep it native for metrics
    # but for display we may resize to original later if needed.
    y_pred_s = theta_hat_s_cand.detach().cpu().numpy()[0]
    return recon_img, y_pred_s

def _plot_latent_exploration_fig(
        z_path,
        cand_clouds,
        cand_values=None,
        value_name="H",
        colorize_candidates=False,
        show_step_labels=True,
        max_step_labels=30,
):
    """
    Returns matplotlib figures: (fig_main, fig_norm, fig_score, fig_cov)
    - best candidate is assumed to be z_path[t+1] for step t
    """
    Zpath = np.asarray(z_path)  # (T+1, D)
    Zpath = np.squeeze(Zpath)
    if Zpath.ndim != 2:
        raise ValueError(f"z_path must become (T+1,D), got {Zpath.shape}")

    T_plus_1, D = Zpath.shape
    T = T_plus_1 - 1
    if len(cand_clouds) != T:
        raise ValueError(f"cand_clouds length must be T={T}, got {len(cand_clouds)}")

    # PCA basis uses path + all clouds
    Z_all = [Zpath]
    for C in cand_clouds:
        Z_all.append(np.asarray(C))
    Z_all = np.concatenate(Z_all, axis=0)

    mean = Z_all.mean(axis=0)  # (D,) IMPORTANT: no keepdims
    Zc = Z_all - mean
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    W = Vt[:2].T  # (D,2)

    Zp2 = (Zpath - mean) @ W  # (T+1,2)

    # ---------- main fig ----------
    fig_main = plt.figure(figsize=(7.5, 6.5))
    ax = fig_main.add_subplot(111)
    mappable = None

    if colorize_candidates:
        if cand_values is None or len(cand_values) != T:
            raise ValueError("colorize_candidates=True requires cand_values with length T")
        vmin = min(float(np.nanmin(v)) for v in cand_values)
        vmax = max(float(np.nanmax(v)) for v in cand_values)

    for t, C in enumerate(cand_clouds):
        C = np.asarray(C)
        C2 = (C - mean) @ W

        if colorize_candidates:
            vals = np.asarray(cand_values[t]).reshape(-1)
            # strict align: must match NUM_CAND
            if vals.size != C2.shape[0]:
                raise ValueError(
                    f"cand_values[{t}] has {vals.size} elems but cand_clouds[{t}] has {C2.shape[0]} points. "
                    "Record one value per candidate (use np.nan for invalid/rejected)."
                )
            mask = np.isfinite(vals)

            # invalid (nan) as faint gray
            if np.any(~mask):
                ax.scatter(C2[~mask, 0], C2[~mask, 1],
                           s=10, alpha=0.08, color="gray", linewidths=0)

            if np.any(mask):
                sc = ax.scatter(C2[mask, 0], C2[mask, 1],
                                c=vals[mask], vmin=vmin, vmax=vmax,
                                s=10, alpha=0.28, linewidths=0)
                mappable = sc
        else:
            ax.scatter(C2[:, 0], C2[:, 1], s=10, alpha=0.18, color="gray", linewidths=0)

        # best = path next point
        z_best = Zpath[t + 1]  # (D,)
        z_best2 = (z_best - mean) @ W  # (2,)
        ax.scatter(
            float(z_best2[0]), float(z_best2[1]),
            s=90, marker="*", color="gold",
            edgecolors="black", linewidths=0.7, zorder=5
        )

    # accepted path
    ax.plot(Zp2[:, 0], Zp2[:, 1], "-o", linewidth=1.6, markersize=4, label="accepted path", zorder=4)
    for i in range(len(Zp2) - 1):
        ax.annotate(
            "",
            xy=(Zp2[i + 1, 0], Zp2[i + 1, 1]),
            xytext=(Zp2[i, 0], Zp2[i, 1]),
            arrowprops=dict(arrowstyle="->", lw=0.8),
            zorder=4
        )

    if show_step_labels:
        stride = max(1, T_plus_1 // max_step_labels)
        for i in range(0, T_plus_1, stride):
            ax.text(Zp2[i, 0], Zp2[i, 1], str(i), fontsize=8)

    ax.set_title("Latent exploration (PCA 2D) with candidate clouds")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")

    if mappable is not None:
        cb = fig_main.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(value_name)

    fig_main.tight_layout()

    # ---------- aux figs ----------
    steps = np.arange(T_plus_1)

    fig_norm = plt.figure(figsize=(7, 4))
    axn = fig_norm.add_subplot(111)
    axn.plot(steps, np.linalg.norm(Zpath, axis=1))
    axn.set_title("||z|| over accepted steps")
    axn.set_xlabel("step")
    axn.set_ylabel("||z||")
    fig_norm.tight_layout()

    fig_score = None
    fig_cov = None
    return fig_main, fig_norm

with tab5:
    st.header("Heuristic Latent Space Exploration")

    def _compute_metrics(recon_img_rgb: np.ndarray):
        """Use channel-0 for dendrite metrics; returns (analysis_fig, metrics_dict, scores_dict)."""
        # generate_analysis_figure expects a 2D gray image in your other tabs
        result_fig, metrics, scores = generate_analysis_figure(recon_img_rgb[..., 0])
        return result_fig, metrics, scores

    # -----------------------------
    # UI: seed selection
    # -----------------------------
    st.subheader("Initial state")

    seed_mode = st.radio(
        "Choose initial state source",
        ["Default", "Upload image", "From test images"],
        index=0,  # ✅ 默认选 Synthetic
        horizontal=True,
        key="tab5_seed_mode"
    )

    seed_image = None
    seed_name = None
    if seed_mode == "Default":
        seed_name = "Default"
        # ---- step 1: generate 50x50 base image ----
        base_eta = np.zeros((50, 50), dtype=np.float32)
        base_eta[:, :5] = 1.0  # 左侧 5 列为 1

        base_c = np.zeros((50, 50), dtype=np.float32)
        base_c[:, :5] = 0.2
        base_c[:, 5:] = 0.8

        base_p = np.zeros((50, 50), dtype=np.float32)

        # ---- step 2: expand to 3 channels ----
        seed_image = np.stack([base_eta, base_c, base_p], axis=-1)  # (50, 50, 3)

    elif seed_mode == "Upload":
        up_seed = st.file_uploader(
            "Upload a seed image (.npy or image file). This seed is only used to get an initial latent z.",
            type=["npy", "jpg", "png", "jpeg", "bmp", "tiff"],
            key="tab5_seed_uploader",
        )
        if up_seed is not None:
            if up_seed.name.endswith(".npy"):
                buf = io.BytesIO(up_seed.getvalue())
                seed_image = np.load(buf)
            else:
                seed_image = np.array(Image.open(up_seed).convert("RGB")) / 255.0
            seed_name = up_seed.name
    else:
        if test_images:
            test_names = [p.name for p in test_images]
            seed_name = st.selectbox("Choose a seed from test images", test_names, key="tab5_seed_select")
            if seed_name:
                name_to_path = {p.name: p for p in test_images}
                seed_image = load_image_from_path(name_to_path[seed_name])
        else:
            st.warning("No test images available.")

    st.markdown("---")

    # -----------------------------
    # UI: exploration hyperparameters
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        STEPS_UI = st.number_input("Steps", min_value=1, max_value=500, value=60, step=1, key="tab5_steps")
    with c2:
        NUM_CAND_UI = st.number_input("NUM_CAND", min_value=4, max_value=256, value=48, step=4, key="tab5_num_cand")
    with c3:
        RW_SIGMA_UI = st.slider("RW_SIGMA", 0.01, 2.0, 0.25, 0.01, key="tab5_sigma")
    with c4:
        enforce_color = st.checkbox("Colorize candidates by H", value=True, key="tab5_colorize")

    st.caption("H = -||params(z_cand) - params(z_current)|| - (score_cand - score_current). "
               "Reject if coverage decreases or t decreases.")

    run_btn = st.button("🚀 Run Exploration", type="primary", disabled=(seed_image is None))

    if run_btn and seed_image is not None:
        with st.spinner("Running exploration..."):
            # ---- init from seed image ----
            recon0_rgb, y_pred_s, z = _encode_image_get_z_and_recon(seed_image)

            fig0, metrics0, scores0 = _compute_metrics(recon0_rgb)
            s = float(scores0["empirical_score"])
            c = float(metrics0["dendrite_coverage"])
            t = float(y_pred_s[0])

            # records
            z_path = [z.copy()]
            score_path = [s]
            coverage_path = [c]
            recon_path = [recon0_rgb]  # model-res recon for display
            analysis_figs = [fig0]

            cand_clouds = []
            cand_H = []

            progress = st.progress(0.0)

            for step in range(1, int(STEPS_UI) + 1):
                best_z = None
                best_H_score = -1e18
                best_score = -1e18
                best_recon = None
                best_params = None
                best_coverage = None

                z_cands = np.empty((int(NUM_CAND_UI), z.shape[0]), dtype=np.float32)
                H_list = np.full((int(NUM_CAND_UI),), np.nan, dtype=np.float32)  # nan for rejected/unevaluated

                for i in range(int(NUM_CAND_UI)):
                    dz = np.random.randn(*z.shape).astype(np.float32) * float(RW_SIGMA_UI)
                    z_cand = z + dz
                    z_cands[i] = z_cand

                    recon_cand_rgb, y_pred_s_cand = _inference_from_z(z_cand)
                    _, metrics_cand, scores_cand = _compute_metrics(recon_cand_rgb)

                    s_cand = float(scores_cand["empirical_score"])
                    c_cand = float(metrics_cand["dendrite_coverage"])
                    t_cand = float(y_pred_s_cand[0])

                    # same heuristic as your explore script
                    H = - float(np.linalg.norm(y_pred_s_cand - y_pred_s)) - (s_cand - s)
                    H_list[i] = H

                    # reject constraints
                    if (c_cand < c) or (t_cand < t):
                        continue

                    if H > best_H_score:
                        best_H_score = H
                        best_score = s_cand
                        best_z = z_cand
                        best_recon = recon_cand_rgb
                        best_params = y_pred_s_cand
                        best_coverage = c_cand

                cand_clouds.append(z_cands.copy())
                cand_H.append(H_list.copy())

                if best_z is None:
                    st.warning(f"Stopped early at step={step}: no valid candidate (all rejected).")
                    break

                # accept best
                z = best_z
                y_pred_s = best_params
                s = float(best_score)
                c = float(best_coverage)
                t = float(best_params[0])

                z_path.append(z.copy())
                score_path.append(s)
                coverage_path.append(c)
                recon_path.append(best_recon)

                fig_step, _, _ = _compute_metrics(best_recon)
                analysis_figs.append(fig_step)

                progress.progress(step / float(STEPS_UI))

            progress.empty()

        # -----------------------------
        # Display results
        # -----------------------------
        st.success(f"✅ Finished. Accepted steps: {len(z_path) - 1}")

        st.subheader("🧭 Latent exploration visualization")
        fig_main, fig_norm = _plot_latent_exploration_fig(
            z_path=z_path,
            cand_clouds=cand_clouds,
            cand_values=cand_H,
            value_name="H",
            colorize_candidates=bool(enforce_color),
        )
        st.pyplot(fig_main)
        st.pyplot(fig_norm)

        # score / coverage curves
        st.subheader("📈 Score / Coverage over accepted steps")
        df_curves = pd.DataFrame({
            "step": np.arange(len(score_path)),
            "score": score_path,
            "coverage": coverage_path,
            "z_norm": np.linalg.norm(np.asarray(z_path), axis=1),
        }).set_index("step")
        st.line_chart(df_curves[["score", "coverage", "z_norm"]])

        # show a compact gallery: first, last, and a few intermediates
        st.subheader("🖼️ Reconstructions along the accepted path")
        show_idx = list(dict.fromkeys(
            [0, min(1, len(recon_path) - 1), len(recon_path) // 4, len(recon_path) // 2, (3 * len(recon_path)) // 4,
             len(recon_path) - 1]
        ))
        cols = st.columns(len(show_idx))
        for col, idx in zip(cols, show_idx):
            with col:
                st.markdown(f"**step {idx}**")
                show_coolwarm(recon_path[idx][..., 0], caption=f"recon step {idx}")

        st.subheader("🔍 Detailed analysis figures (optional)")
        if st.checkbox("Show analysis figure for each accepted step (can be slow)", value=False, key="tab5_show_all"):
            for i, fig in enumerate(analysis_figs):
                with st.container(border=True):
                    st.markdown(f"**Accepted step {i}**  · score={score_path[i]:.4f} · coverage={coverage_path[i]:.4f}")
                    st.pyplot(fig)

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
