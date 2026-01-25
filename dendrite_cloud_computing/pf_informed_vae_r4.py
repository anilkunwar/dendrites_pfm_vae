import hashlib
import os
import io
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors, cm
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import streamlit as st

from src.modelv11 import mdn_point_and_confidence
from src.evaluate_metrics import generate_analysis_figure
from src.dataloader import inv_scale_params, smooth_scale, inv_smooth_scale, PARAM_RANGES
from src.helper import *

def ensure_valid_image(image):
    """
    Ensure image is a valid 2D numpy array for display
    """
    if image is None:
        return None
    
    # Convert to numpy array if not already
    if not isinstance(image, np.ndarray):
        try:
            image = np.array(image)
        except Exception as e:
            return None
    
    # Handle different dimensionalities
    if image.ndim == 3:
        # If RGB or RGBA, take first channel
        if image.shape[2] >= 1:
            image = image[:, :, 0]
    elif image.ndim > 3:
        # For higher dimensional tensors, take first slice
        slices = [0] * (image.ndim - 2)
        image = image[tuple([0, 0] + slices)]
    elif image.ndim < 2:
        return None
    
    # Ensure the image is float type for proper normalization
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
    
    # Handle NaN or Inf values
    if np.isnan(image).any() or np.isinf(image).any():
        # Replace NaN with min value, Inf with max value
        min_val = np.nanmin(image[np.isfinite(image)])
        max_val = np.nanmax(image[np.isfinite(image)])
        image = np.nan_to_num(image, nan=min_val, posinf=max_val, neginf=min_val)
    
    # Ensure image has some variation for visualization
    if np.allclose(image.min(), image.max(), atol=1e-7):
        # Add small random noise if image is uniform
        image = image + np.random.normal(0, 1e-4, image.shape)
    
    return image

def show_coolwarm(gray_image, caption, container=None):
    """
    Display image using coolwarm colormap with robust error handling
    """
    # Ensure image is valid before proceeding
    gray_image = ensure_valid_image(gray_image)
    if gray_image is None:
        st.warning("Invalid image data for visualization")
        return
    
    try:
        # Safely compute min and max values
        vmin = float(gray_image.min())
        vmax = float(gray_image.max())
        
        # Handle case where min and max are too close
        if abs(vmax - vmin) < 1e-7:
            mid = (vmax + vmin) / 2
            vmin = mid - 0.5
            vmax = mid + 0.5
        
        # Create normalization and colormap
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        colored_img = cm.coolwarm(norm(gray_image))
        
        # Display the image
        if container is None:
            st.image(colored_img, caption=caption, use_container_width=True)
        else:
            container.image(colored_img, caption=caption, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        # Try a simpler display method as fallback
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(gray_image, cmap='coolwarm')
            ax.set_title(f"Fallback view: {caption}")
            fig.colorbar(im, ax=ax)
            if container is None:
                st.pyplot(fig)
            else:
                container.pyplot(fig)
            plt.close(fig)
        except:
            pass

def analyze_image(image, image_name:str):
    # Display original image (without preprocessing)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Original Image (Only 1st channel)")
        show_coolwarm(image[..., 0], caption=f"Selected: {image_name}")
        st.caption(
            f"Size: {image.shape[0]}×{image.shape[1]}, Max value: {np.max(image):.2f}, Min value: {np.min(image):.2f}")

    # Process image
    recon_image, ctr_array, conf_s, conf_global_s = process_image(image, model, expected_size, var_scale)

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

def _plot_latent_exploration_tsne(
    z_path,
    cand_clouds,
    cand_values=None,
    value_name="H",
    colorize_candidates=False,
    show_step_labels=True,
    max_step_labels=30,
    hopping_strengths=None,
    random_state=42
):
    """
    Creates a t-SNE visualization of latent space exploration with color-coded hopping strengths
    """
    Zpath = np.asarray(z_path)
    Zpath = np.squeeze(Zpath)
    if Zpath.ndim != 2:
        raise ValueError(f"z_path must become (T+1,D), got {Zpath.shape}")

    T_plus_1, D = Zpath.shape
    T = T_plus_1 - 1
    
    # Collect all points for t-SNE
    all_points = [Zpath]
    point_types = ['path'] * (T + 1)  # Mark path points
    step_indices = list(range(T + 1))  # Step indices for path points
    
    # Add candidate points with metadata
    for t, cloud in enumerate(cand_clouds):
        cloud = np.asarray(cloud)
        all_points.append(cloud)
        point_types.extend(['candidate'] * len(cloud))
        step_indices.extend([t] * len(cloud))
    
    # Combine all points
    Z_all = np.concatenate(all_points, axis=0)
    point_types = np.array(point_types)
    step_indices = np.array(step_indices)
    
    # Compute t-SNE embedding
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(Z_all)-1))
    Z_embedded = tsne.fit_transform(Z_all)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color mapping for hopping strengths
    if hopping_strengths is not None:
        # Normalize hopping strengths for color mapping
        norm = plt.Normalize(min(hopping_strengths), max(hopping_strengths))
        cmap = plt.cm.viridis
        
        # Plot path points with color gradient based on step
        path_colors = [cmap(norm(hopping_strengths[i])) for i in range(T+1)]
        ax.scatter(Z_embedded[:T+1, 0], Z_embedded[:T+1, 1], 
                  c=path_colors, s=80, marker='o', edgecolor='k', 
                  label='Exploration Path', zorder=10)
        
        # Add colorbar for hopping strengths
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Hopping Strength (σ)', fontsize=12)
    else:
        # Default path plotting without hopping strength
        ax.plot(Z_embedded[:T+1, 0], Z_embedded[:T+1, 1], '-o', 
                linewidth=2, markersize=8, color='red', label='Path', zorder=10)
    
    # Plot candidate clouds
    if colorize_candidates and cand_values is not None:
        candidate_start = T + 1
        candidate_values = []
        candidate_indices = []
        
        for t, cloud in enumerate(cand_clouds):
            n_candidates = len(cloud)
            start_idx = candidate_start + sum(len(c) for c in cand_clouds[:t])
            end_idx = start_idx + n_candidates
            
            values = cand_values[t]
            if len(values) != n_candidates:
                values = np.full(n_candidates, np.nan)
            
            valid_mask = ~np.isnan(values)
            ax.scatter(Z_embedded[start_idx:end_idx][valid_mask, 0], 
                      Z_embedded[start_idx:end_idx][valid_mask, 1],
                      c=values[valid_mask], s=15, alpha=0.6, 
                      cmap='coolwarm', vmin=np.min(values), vmax=np.max(values))
            
            # Store for colorbar
            candidate_values.extend(values[valid_mask])
            candidate_indices.extend(range(start_idx, end_idx))
        
        if candidate_values:
            sm = plt.cm.ScalarMappable(cmap='coolwarm', 
                                     norm=plt.Normalize(vmin=min(candidate_values), 
                                                       vmax=max(candidate_values)))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.08)
            cbar.set_label(value_name, fontsize=12)
    else:
        # Plot all candidates in gray
        candidate_mask = (point_types == 'candidate')
        ax.scatter(Z_embedded[candidate_mask, 0], Z_embedded[candidate_mask, 1],
                  s=10, alpha=0.3, color='gray', label='Candidates')
    
    # Annotate path steps
    if show_step_labels:
        stride = max(1, T_plus_1 // max_step_labels)
        for i in range(0, T_plus_1, stride):
            ax.text(Z_embedded[i, 0], Z_embedded[i, 1], str(i), 
                   fontsize=9, fontweight='bold', zorder=15)
    
    # Highlight start and end points
    ax.scatter(Z_embedded[0, 0], Z_embedded[0, 1], s=150, marker='*', 
              color='gold', edgecolor='black', linewidth=1.5, zorder=20, label='Start')
    ax.scatter(Z_embedded[T, 0], Z_embedded[T, 1], s=150, marker='X', 
              color='darkred', edgecolor='black', linewidth=1.5, zorder=20, label='End')
    
    ax.set_title('Latent Space Exploration (t-SNE)', fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    return fig

def analyze_parameter_importance(df_history):
    """
    Analyzes the importance of control parameters on interfacial morphology metrics
    Returns figures and analysis results
    """
    # Extract relevant columns
    param_cols = [col for col in df_history.columns if col not in ['step', 'score', 'coverage', 'z_norm', 't']]
    metrics = ['score', 'coverage']
    
    # Prepare data
    X = df_history[param_cols].values
    y_score = df_history['score'].values
    y_coverage = df_history['coverage'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Linear regression for importance
    model_score = LinearRegression().fit(X_scaled, y_score)
    model_coverage = LinearRegression().fit(X_scaled, y_coverage)
    
    # Get coefficients
    coef_score = pd.Series(model_score.coef_, index=param_cols)
    coef_coverage = pd.Series(model_coverage.coef_, index=param_cols)
    
    # Calculate correlation coefficients
    corr_score = {}
    corr_coverage = {}
    pval_score = {}
    pval_coverage = {}
    
    for param in param_cols:
        corr_s, p_s = pearsonr(df_history[param], df_history['score'])
        corr_c, p_c = pearsonr(df_history[param], df_history['coverage'])
        corr_score[param] = corr_s
        pval_score[param] = p_s
        corr_coverage[param] = corr_c
        pval_coverage[param] = p_c
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Plot 1: Regression coefficients
    sorted_idx = np.argsort(np.abs(coef_score))[::-1]
    top_params = [param_cols[i] for i in sorted_idx[:10]]
    
    axes[0].barh(top_params, coef_score[top_params], color='skyblue', edgecolor='navy')
    axes[0].set_title('Parameter Impact on Dendrite Score', fontsize=14)
    axes[0].set_xlabel('Regression Coefficient', fontsize=12)
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: Correlation with coverage
    sorted_idx_cov = np.argsort(np.abs(list(corr_coverage.values())))[::-1]
    top_params_cov = [list(corr_coverage.keys())[i] for i in sorted_idx_cov[:10]]
    
    bar_colors = ['green' if corr_coverage[p] > 0 else 'red' for p in top_params_cov]
    axes[1].barh(top_params_cov, [corr_coverage[p] for p in top_params_cov], 
                color=bar_colors, edgecolor='darkgreen')
    axes[1].set_title('Parameter Correlation with Coverage', fontsize=14)
    axes[1].set_xlabel('Pearson Correlation', fontsize=12)
    axes[1].axvline(0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Create summary table
    importance_df = pd.DataFrame({
        'Parameter': param_cols,
        'Score_Coefficient': coef_score.values,
        'Coverage_Correlation': [corr_coverage[p] for p in param_cols],
        'Score_PValue': [pval_score[p] for p in param_cols],
        'Coverage_PValue': [pval_coverage[p] for p in param_cols]
    })
    
    # Sort by absolute impact
    importance_df['Combined_Importance'] = (
        np.abs(importance_df['Score_Coefficient']) + 
        np.abs(importance_df['Coverage_Correlation'])
    )
    importance_df = importance_df.sort_values('Combined_Importance', ascending=False)
    
    return fig, importance_df

st.set_page_config(layout="wide", page_title="VAE Image Reconstruction")

st.title("🎨 VAE Image Reconstruction & Analysis")
st.markdown("Upload an image or select from test images to reconstruct it and analyze predicted control parameters.")

# Load model
device = "cpu"
# Sidebar for model info and controls
with st.sidebar:
    st.header("⚙️ Controls & Information")

    st.markdown("### Model Information")
    st.info("""
    This VAE model:
    - Reconstructs 256×256 RGB images
    - Predicts 15 control parameters
    - Uses a multi-kernel residual architecture
    """)

    # check for models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "knowledge_base")
    model_paths = {p: os.path.join(model_dir, p) for p in os.listdir(model_dir)}
    selected_model_name = st.selectbox("Choose a model:", list(model_paths.keys()))
    model = load_model(os.path.join(model_dir, selected_model_name), device)

    # Check for test images
    test_folder, test_images = get_test_images()
    test_names = [p.name for p in test_images]

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
        selected_image_name = st.selectbox("Choose a test image:", test_names, index=0)

        if selected_image_name:
            # Find the selected image path
            selected_idx = test_names.index(selected_image_name)
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
            options=test_names,
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
                        recon_img, ctr_array, conf_s, conf_global_s = process_image(image, model, expected_size, var_scale)

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

def file_fingerprint(data) -> str:
    h = hashlib.sha256(data).hexdigest()
    return h
def tab4_add_item(img: np.ndarray, name:str, id:str, source: str):
    st.session_state.tab4_items.append({
        "id": id,
        "name": name,
        "origin": img,
        "source": source,
        "result": None,
        "score": None,
    })
with tab4:
    st.header("Dendrite Intensity Analysis")

    # ========== 1) Session state for tab4 ==========
    if st.session_state.get("tab4_items") is None:
        st.session_state.tab4_items = []
    past_files = [item["id"] for item in st.session_state.tab4_items] # which files are not here now?

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
        if up_files:
            for uf in up_files:
                fid = file_fingerprint(uf.getvalue())
                if fid in past_files:
                    past_files.remove(fid)
                    continue
                if uf.name.endswith(".npy"):
                    buf = io.BytesIO(uf.getvalue())
                    img = np.load(buf)
                else:
                    img = np.array(Image.open(uf).convert("RGB")) / 255.0
                tab4_add_item(img, uf.name, fid, source="upload")
        st.caption("Tip: you can upload multiple times. Newly uploaded images will be added to the list below.")
    with right_col:
        st.subheader("Select from Test Images")
        if test_images:
            selected_dendrite_images = st.multiselect(
                "Select images for analysis:",
                options=test_names,
                default=[]
            )
            if selected_dendrite_images:
                name_to_path = {p.name: p for p in test_images}
                for nm in selected_dendrite_images:
                    fid = file_fingerprint(str(name_to_path[nm]).encode("utf-8"))
                    if fid in past_files:
                        past_files.remove(fid)
                        continue
                    img = load_image_from_path(name_to_path[nm])
                    tab4_add_item(img, nm, fid, source="test")
        else:
            st.warning("No test images found for analysis.")
    # delete file that no longer here
    for item in st.session_state.tab4_items.copy():
        if item["id"] in past_files:
            st.session_state.tab4_items.remove(item)
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
                    if item["result"] is None:
                        result_img, _, scores = generate_analysis_figure(np.clip(item["origin"][..., 0], 0, 1))
                        item["result"] = result_img
                        item["score"] = scores["empirical_score"]
                    else:
                        st.info("Use old results for analysis.")
                    header_cols = st.columns([1, 1])
                    with header_cols[0]:
                        st.markdown(f"**{item['name']}**  · from：`{item['source']}`")
                    with header_cols[1]:
                        st.metric("Score", f"{item['score']:.4f}")
                    st.pyplot(item["result"])
                progress_bar.progress((idx + 1) / len(st.session_state.tab4_items))
        st.success(f"✅ Processed {len(st.session_state.tab4_items)} images")

def _params_to_table(y_pred: np.ndarray, confidence, extra: dict):
    rows = [{"name": f"{param_names[i]}", "value": float(y_pred[i]), f"confidence under {var_scale}": confidence[i]} for i in range(len(y_pred))]
    for k, v in extra.items():
        rows.append({"name": k, "value": float(v), f"confidence under {var_scale}": -1})
    return pd.DataFrame(rows)

def _update_live(step_i: int,
                 recon_rgb: np.ndarray,
                 z_1d: np.ndarray,
                 y_pred_s: np.ndarray,
                 y_pred_conf: np.ndarray,
                 score: float,
                 coverage: float,
                 cand_clouds=None, cand_H=None):
    """
    Append history + refresh the Live viewer + refresh metrics table + refresh candidate summary.
    Safe for repeated calls.
    """
    hist = st.session_state.explore_hist

    hist["step"].append(int(step_i))
    hist["recon"].append(recon_rgb)
    hist["z"].append(z_1d)
    hist["params"].append(y_pred_s)
    hist["params_confidence"].append(y_pred_conf)
    hist["score"].append(float(score))
    hist["coverage"].append(float(coverage))
    if cand_clouds is not None:
        hist["cand_clouds"].append(cand_clouds)
    if cand_H is not None:
        hist["cand_H"].append(cand_H)

    # auto-jump viewer to latest step
    st.session_state["tab5_view_step"] = len(hist["step"]) - 1

    # update live image
    show_coolwarm(recon_rgb, f"Step {step_i} (latest accepted)", live_img_placeholder)

    # update metrics table (latest)
    df = _params_to_table(y_pred_s, y_pred_conf, {
        "score": float(score),
        "coverage": float(coverage),
        "||z||": float(np.linalg.norm(np.asarray(z_1d).reshape(-1))),
    })
    metrics_placeholder.dataframe(df, use_container_width=True, hide_index=True)

    log_container.code(
        f"step={step_i} score={score:.3f} t={y_pred_s[0]:.3f}, Coverage={coverage:.3f}, ||z||={np.linalg.norm(z):.2f}",
        language="text"
    )

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
    Returns matplotlib figure: fig_main
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

    mean = Z_all.mean(axis=0)  # (D,)
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
            if vals.size != C2.shape[0]:
                raise ValueError(
                    f"cand_values[{t}] has {vals.size} elems but cand_clouds[{t}] has {C2.shape[0]} points. "
                    "Record one value per candidate (use np.nan for invalid/rejected)."
                )
            mask = np.isfinite(vals)

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

    return fig_main

with tab5:
    st.header("Heuristic Latent Space Exploration")

    st.subheader("Initial state")
    seed_mode = st.radio(
        "Choose initial state source",
        ["Default", "Upload image", "From test images"],
        index=0,  # ✅ default select Default
        horizontal=True,
        key="tab5_seed_mode"
    )

    seed_image = None
    seed_name = None
    if seed_mode == "Default":
        seed_name = "Default"
        # ---- step 1: generate 50x50 base image ----
        base_eta = np.zeros((50, 50), dtype=np.float32)
        base_eta[:, :5] = 1.0  # left 5 columns as 1

        base_c = np.zeros((50, 50), dtype=np.float32)
        base_c[:, :5] = 0.2
        base_c[:, 5:] = 0.8

        base_p = np.zeros((50, 50), dtype=np.float32)

        # ---- step 2: expand to 3 channels ----
        seed_image = np.stack([base_eta, base_c, base_p], axis=-1)  # (50, 50, 3)
    elif seed_mode == "Upload image":
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
            seed_name = st.selectbox("Choose a seed from test images", test_names, key="tab5_seed_select")
            if seed_name:
                name_to_path = {p.name: p for p in test_images}
                seed_image = load_image_from_path(name_to_path[seed_name])
        else:
            st.warning("No test images available.")

    st.markdown("---")

    # -----------------------------
    # UI: Parameter Bias Weights Configuration
    # -----------------------------
    st.subheader("PropertyParams Configuration")
    
    # Set parameter weights based on step 12 values from CSV file
    # step,score,coverage,hopping_strength,t,POT_LEFT,fo,Al,Bl,Cl,As,Bs,Cs,cleq,cseq,L1o,L2o,ko,Noise
    # 12,8.543683990691765,0.3723958333333333,0.1662757831681574,0.5105248689651489,0.34250542521476746,0.43922367691993713,0.7341154217720032,1.1102296113967896,0.7483778595924377,0.7128080725669861,0.7358441352844238,0.7393836975097656,0.46837639808654785,0.8202276825904846,0.877817690372467,0.9840682744979858,0.7861328125,0.6163419485092163
    
    step12_weights = {
        "t": 0.5105248689651489,
        "POT_LEFT": 0.34250542521476746,
        "fo": 0.43922367691993713,
        "Al": 0.7341154217720032,
        "Bl": 1.1102296113967896,
        "Cl": 0.7483778595924377,
        "As": 0.7128080725669861,
        "Bs": 0.7358441352844238,
        "Cs": 0.7393836975097656,
        "cleq": 0.46837639808654785,
        "cseq": 0.8202276825904846,
        "L1o": 0.877817690372467,
        "L2o": 0.9840682744979858,
        "ko": 0.7861328125,
        "Noise": 0.6163419485092163
    }
    
    # Initialize default parameter weights if not in session state
    if "param_weights" not in st.session_state:
        st.session_state.param_weights = {}
        # Set weights based on step 12 values
        for param in param_names:
            if param in step12_weights:
                st.session_state.param_weights[param] = step12_weights[param]
            else:
                st.session_state.param_weights[param] = 1.0
    
    # Display parameter weights configuration
    st.markdown("### Parameter Bias Weights")
    st.caption("Assign weights to each parameter to influence their importance during exploration. Higher weights mean the parameter changes more significantly affect the exploration path. Default values are from step 12 of the CSV file - dendrites-attributes.csv.")
    
    # Create columns for parameter weight inputs
    weight_cols = st.columns(3)
    weight_col_idx = 0
    
    # Create input fields for each parameter weight
    for i, param in enumerate(param_names):
        with weight_cols[weight_col_idx]:
            current_weight = st.session_state.param_weights.get(param, 1.0)
            new_weight = st.number_input(
                f"Weight for {param}",
                min_value=-5.0, 
                max_value=5.0, 
                value=float(current_weight), 
                step=0.1,
                key=f"weight_{param}"
            )
            st.session_state.param_weights[param] = new_weight
        
        weight_col_idx = (weight_col_idx + 1) % len(weight_cols)
    
    # Add options for weight presets
    st.markdown("### Weight Presets")
    preset_option = st.selectbox(
        "Apply weight preset:",
        ["Custom (CSV file values)", "All Equal", "Emphasize t", "Emphasize Physical Parameters", "Random Weights"],
        index=0,
        key="weight_preset"
    )
    
    if st.button("Apply Preset"):
        if preset_option == "Custom (Step 12 CSV values)":
            # Reset to step 12 values
            for param in param_names:
                if param in step12_weights:
                    st.session_state.param_weights[param] = step12_weights[param]
                else:
                    st.session_state.param_weights[param] = 1.0
        elif preset_option == "All Equal":
            st.session_state.param_weights = {param: 1.0 for param in param_names}
        elif preset_option == "Emphasize t":
            st.session_state.param_weights = {param: 1.0 for param in param_names}
            st.session_state.param_weights["t"] = 3.0
        elif preset_option == "Emphasize Physical Parameters":
            # Physical parameters typically include things like diffusion coefficients, interface energies, etc.
            physical_params = ["diff", "kappa_eta", "M", "kappa_c"]  # Example physical parameters
            st.session_state.param_weights = {param: 1.0 for param in param_names}
            for pp in physical_params:
                if pp in st.session_state.param_weights:
                    st.session_state.param_weights[pp] = 2.5
        elif preset_option == "Random Weights":
            np.random.seed(42)  # For reproducibility
            st.session_state.param_weights = {param: round(np.random.uniform(-2, 2), 1) for param in param_names}
        
        st.success(f"Applied preset: {preset_option}")
        st.rerun()
    
    # Visualize parameter weights
    st.markdown("### Weight Visualization")
    weights_df = pd.DataFrame({
        "Parameter": list(st.session_state.param_weights.keys()),
        "Weight": list(st.session_state.param_weights.values())
    })
    
    # Create a bar chart
    fig_weights, ax_weights = plt.subplots(figsize=(10, 4))
    bar_colors = ['red' if w < 0 else 'green' for w in weights_df['Weight']]
    bars = ax_weights.bar(weights_df['Parameter'], weights_df['Weight'], color=bar_colors)
    ax_weights.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_weights.set_title('Parameter Bias Weights')
    ax_weights.set_ylabel('Weight Value')
    ax_weights.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax_weights.annotate(f'{height:.1f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig_weights)
    
    # Display the weights table
    st.dataframe(weights_df.style.format({"Weight": "{:.2f}"}).background_gradient(subset=["Weight"], cmap="coolwarm"))

    st.markdown("---")

    # -----------------------------
    # UI: exploration hyperparameters
    # -----------------------------
    st.subheader("Exploration Configuration")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        STEPS_UI = st.number_input("Steps", min_value=1, max_value=500, value=30, step=1)
    with c2:
        NUM_CAND_UI = st.number_input("Candidates/step", min_value=4, max_value=256, value=32, step=4)
    with c3:
        # Define default RW_SIGMA_UI outside the conditional block
        RW_SIGMA_UI = 0.25
        hopping_strengths = [RW_SIGMA_UI] * (STEPS_UI + 1)
        
        # Modified to support multiple hopping strengths including functional options
        HOPPING_MODE = st.selectbox("Hopping Mode", 
                                   ["Single Strength", "Multiple Strengths", "Functional Decreasing", "Functional Increasing"],
                                   index=0)
        
        if HOPPING_MODE == "Single Strength":
            RW_SIGMA_UI = st.slider("Hopping Strength (σ)", 0.01, 2.0, 0.25, 0.01)
            hopping_strengths = [RW_SIGMA_UI] * (STEPS_UI + 1)
        elif HOPPING_MODE == "Multiple Strengths":
            MIN_SIGMA = st.slider("Min Strength", 0.01, 1.0, 0.05, 0.01)
            MAX_SIGMA = st.slider("Max Strength", 0.1, 2.0, 0.5, 0.01)
            # Create linear progression of hopping strengths
            hopping_strengths = np.linspace(MIN_SIGMA, MAX_SIGMA, STEPS_UI + 1)
            st.caption(f"Strengths range from {MIN_SIGMA:.2f} to {MAX_SIGMA:.2f}")
        elif HOPPING_MODE == "Functional Decreasing":
            MAX_SIGMA_FUNC = st.slider("Starting Strength (max)", 0.1, 5.0, 2.0, 0.1)
            MIN_SIGMA_FUNC = st.slider("Ending Strength (min)", 0.01, 1.0, 0.05, 0.01)
            DECAY_MODE = st.selectbox("Decay Function", ["Exponential", "Linear", "Logarithmic"], index=0)
            
            if DECAY_MODE == "Exponential":
                DECAY_RATE = st.slider("Decay Rate", 0.1, 5.0, 1.0, 0.1)
                # Create exponential decay from max to min
                t = np.linspace(0, 1, STEPS_UI + 1)
                hopping_strengths = MIN_SIGMA_FUNC + (MAX_SIGMA_FUNC - MIN_SIGMA_FUNC) * np.exp(-DECAY_RATE * t)
            elif DECAY_MODE == "Linear":
                # Linear decrease from max to min
                hopping_strengths = np.linspace(MAX_SIGMA_FUNC, MIN_SIGMA_FUNC, STEPS_UI + 1)
            else:  # Logarithmic
                # Logarithmic decrease from max to min
                t = np.linspace(0, 1, STEPS_UI + 1)
                hopping_strengths = MIN_SIGMA_FUNC + (MAX_SIGMA_FUNC - MIN_SIGMA_FUNC) * (1 - np.log(1 + 9*t) / np.log(10))
            
            st.caption(f"Strengths decrease from {MAX_SIGMA_FUNC:.2f} to {MIN_SIGMA_FUNC:.2f} using {DECAY_MODE} decay")
        else:  # Functional Increasing
            MIN_SIGMA_FUNC_INC = st.slider("Starting Strength (min)", 0.01, 1.0, 0.05, 0.01)
            MAX_SIGMA_FUNC_INC = st.slider("Ending Strength (max)", 0.1, 5.0, 2.0, 0.1)
            GROWTH_MODE = st.selectbox("Growth Function", ["Exponential", "Linear", "Logarithmic"], index=0)
            
            if GROWTH_MODE == "Exponential":
                GROWTH_RATE = st.slider("Growth Rate", 0.1, 5.0, 1.0, 0.1)
                # Create exponential growth from min to max
                t = np.linspace(0, 1, STEPS_UI + 1)
                hopping_strengths = MIN_SIGMA_FUNC_INC + (MAX_SIGMA_FUNC_INC - MIN_SIGMA_FUNC_INC) * (1 - np.exp(-GROWTH_RATE * t))
            elif GROWTH_MODE == "Linear":
                # Linear increase from min to max
                hopping_strengths = np.linspace(MIN_SIGMA_FUNC_INC, MAX_SIGMA_FUNC_INC, STEPS_UI + 1)
            else:  # Logarithmic
                # Logarithmic growth from min to max
                t = np.linspace(0, 1, STEPS_UI + 1)
                hopping_strengths = MIN_SIGMA_FUNC_INC + (MAX_SIGMA_FUNC_INC - MIN_SIGMA_FUNC_INC) * (np.log(1 + 9*t) / np.log(10))
            
            st.caption(f"Strengths increase from {MIN_SIGMA_FUNC_INC:.2f} to {MAX_SIGMA_FUNC_INC:.2f} using {GROWTH_MODE} growth")
    with c4:
        STRICT_UI = st.checkbox("Strict mode", value=False)
    with c5:
        TSNE_PERPLEXITY = st.slider("t-SNE Perplexity", 5, 100, 30, 5)

    st.caption("H = -||weighted_params(z_cand) - weighted_params(z_current)|| - (score_cand - score_current). "
               "Reject if coverage decreases or t decreases.")

    run_btn = st.button("🚀 Run Exploration", type="primary", disabled=(seed_image is None))

    st.markdown("---")

    # -----------------------------
    # TAB5: Live viewer + history + params/score panel
    # -----------------------------
    st.subheader("🖼️ Live Exploration Viewer")
    if st.session_state.get("explore_hist") is None:
        st.session_state.explore_hist = {
            "recon": [],  # list of (H,W,3) float
            "analysis_fig": [],  # list of matplotlib figs (optional)
            "z": [],  # list of (D,)
            "params": [],  # list of (P,) y_pred_s
            "params_confidence": [],
            "score": [],  # list of float
            "coverage": [],  # list of float
            "step": [],  # list of int
            "cand_clouds": [],
            "cand_H": [],
            "hopping_strength": [],
            "param_weights": []  # Store parameter weights used at each step
        }

    # show results in one step
    log_container = st.container(height=220)

    # show live
    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        # Viewer containers
        live_box = st.container(border=True)
    with right:
        metrics_box = st.container(border=True)

    # ---- Live viewer (current + selected historical) ----
    with live_box:
        st.markdown("### Live")

        live_img_placeholder = st.empty()
        live_caption_placeholder = st.empty()

    # ---- Metrics / params panel for selected step ----
    with metrics_box:
        st.markdown("### Params & Scores")

        metrics_placeholder = st.empty()

    if run_btn and seed_image is not None:

        # clean all the images
        st.session_state.explore_hist = {
            "recon": [],  # list of (H,W,3) float
            "analysis_fig": [],  # list of matplotlib figs (optional)
            "z": [],  # list of (D,)
            "params": [],  # list of (P,) y_pred_s
            "params_confidence": [],
            "score": [],  # list of float
            "coverage": [],  # list of float
            "step": [],  # list of int
            "cand_clouds": [],
            "cand_H": [],
            "hopping_strength": [],
            "param_weights": []  # Store parameter weights used at each step
        }
        st.session_state.tab5_view_step = 0

        with st.spinner("Running exploration..."):
            # Determine latent dimension from model architecture
            latent_dim = None
            if hasattr(model, 'latent_dim'):
                latent_dim = model.latent_dim
            elif hasattr(model, 'decoder') and hasattr(model.decoder, 'fc') and hasattr(model.decoder.fc, 'in_features'):
                latent_dim = model.decoder.fc.in_features
            else:
                # Fallback to shape of z from initial encoding
                seed_image_tensor = prep_tensor_from_image(seed_image, expected_size)
                with torch.no_grad():
                    _, _, _, _, z_test = model(seed_image_tensor)
                latent_dim = z_test.shape[1]
                del z_test
            
            st.info(f"Detected latent dimension: {latent_dim}")
            
            progress_bar = st.progress(0)
            with torch.no_grad():
                seed_image_tensor = prep_tensor_from_image(seed_image, expected_size)
                recon, mu_q, logvar_q, (pi_s, mu_s, log_sigma_s), z = model(seed_image_tensor)
                # ---- stochastic prediction + confidence (from sampled z) ----
                theta_hat_s, conf_param_s, conf_global_s, modes_s = mdn_point_and_confidence(
                    pi_s, mu_s, log_sigma_s, var_scale=var_scale
                )

            # Ensure reconstruction is properly formatted
            recon = inv_smooth_scale(recon)
            recon = recon.cpu().detach().numpy()[0, 0]
            # Make sure recon is a proper 2D array
            if recon.ndim > 2:
                recon = recon[0]  # Take first channel if multi-dimensional
            
            z = z.cpu().detach().numpy()[0]
            y_pred_s = theta_hat_s.detach().cpu().numpy()[0]
            conf_s = conf_param_s.detach().cpu().numpy()[0]
            conf_global_s = conf_global_s.detach().cpu().numpy()[0]

            _, metrics, scores = generate_analysis_figure(np.clip(recon, 0, 1))
            s = scores["empirical_score"]
            c = metrics["dendrite_coverage"]
            t_val = y_pred_s[0]

            # Initialize with first hopping strength
            if HOPPING_MODE == "Single Strength":
                initial_hopping = RW_SIGMA_UI
            else:
                initial_hopping = hopping_strengths[0]
                
            st.session_state.explore_hist["hopping_strength"].append(initial_hopping)
            st.session_state.explore_hist["param_weights"].append(st.session_state.param_weights.copy())

            # Ensure the image is properly formatted before updating live display
            recon_for_display = ensure_valid_image(recon)
            _update_live(0, recon_for_display, z, y_pred_s, conf_s, s, c)
            
            for step in range(1, STEPS_UI + 1):
                # Get current hopping strength based on mode
                current_hopping = hopping_strengths[step % len(hopping_strengths)]
                
                # 生成候选
                best_z = None
                best_H_score = -1e18
                best_score = -1e18
                best_img = None
                best_params = None
                best_params_confidence = None
                best_coverage = None
                z_cands = []
                H_list = []
                
                # Ensure z has correct dimension
                if z.shape[0] != latent_dim:
                    st.warning(f"Reshaping latent vector from {z.shape[0]} to {latent_dim} dimensions")
                    if z.shape[0] > latent_dim:
                        z = z[:latent_dim]
                    else:
                        z_padded = np.zeros(latent_dim)
                        z_padded[:z.shape[0]] = z
                        z = z_padded
                
                # Get current parameter weights
                current_weights = np.array([st.session_state.param_weights[param] for param in param_names])
                
                for cand_idx in range(NUM_CAND_UI):
                    try:
                        dz = np.random.randn(*z.shape).astype(np.float32) * current_hopping
                        z_cand = z + dz
                        
                        # Ensure candidate has correct dimension
                        if z_cand.shape[0] != latent_dim:
                            if z_cand.shape[0] > latent_dim:
                                z_cand = z_cand[:latent_dim]
                            else:
                                z_cand_padded = np.zeros(latent_dim)
                                z_cand_padded[:z_cand.shape[0]] = z_cand
                                z_cand = z_cand_padded
                        
                        # Create tensor with correct shape and device
                        z_cand_tensor = torch.from_numpy(z_cand).float().to(device)
                        if z_cand_tensor.dim() == 1:
                            z_cand_tensor = z_cand_tensor.unsqueeze(0)
                        
                        # Verify shape matches model expectations
                        if z_cand_tensor.shape[1] != latent_dim:
                            st.warning(f"Reshaping candidate tensor from {z_cand_tensor.shape[1]} to {latent_dim} dimensions")
                            if z_cand_tensor.shape[1] > latent_dim:
                                z_cand_tensor = z_cand_tensor[:, :latent_dim]
                            else:
                                z_cand_tensor_padded = torch.zeros(z_cand_tensor.shape[0], latent_dim).to(device)
                                z_cand_tensor_padded[:, :z_cand_tensor.shape[1]] = z_cand_tensor
                                z_cand_tensor = z_cand_tensor_padded
                        
                        with torch.no_grad():
                            # Call model decoder directly with proper interface
                            if hasattr(model, 'decoder'):
                                # For newer model architecture
                                recon_cand = model.decoder(z_cand_tensor)
                                # Get predictions from the model's prediction head
                                if hasattr(model, 'prediction_head'):
                                    params_pred = model.prediction_head(z_cand_tensor)
                                    # Handle MDN output format if needed
                                    if isinstance(params_pred, tuple) and len(params_pred) == 4:
                                        pi_s_cand, mu_s_cand, log_sigma_s_cand, _ = params_pred
                                        theta_hat_s_cand, conf_param_s_cand, conf_global_s_cand, modes_s_cand = mdn_point_and_confidence(
                                            pi_s_cand, mu_s_cand, log_sigma_s_cand, var_scale=var_scale
                                        )
                                    else:
                                        # Simple prediction output
                                        theta_hat_s_cand = params_pred
                                        conf_param_s_cand = torch.ones_like(theta_hat_s_cand) * 0.95
                                        conf_global_s_cand = torch.tensor([0.95])
                                        modes_s_cand = None
                                else:
                                    # Fallback to inference function
                                    recon_cand, (theta_hat_s_cand, conf_param_s_cand, conf_global_s_cand, modes_s_cand) = \
                                        inference(model, z_cand_tensor, var_scale=var_scale)
                            else:
                                # For older model architecture
                                recon_cand, (theta_hat_s_cand, conf_param_s_cand, conf_global_s_cand, modes_s_cand) = \
                                    inference(model, z_cand_tensor, var_scale=var_scale)
                        
                        # Ensure reconstruction is properly formatted for display
                        recon_cand = inv_smooth_scale(recon_cand)
                        recon_cand = recon_cand.cpu().detach().numpy()[0, 0]
                        if recon_cand.ndim > 2:
                            recon_cand = recon_cand[0]  # Take first channel if multi-dimensional
                        
                        y_pred_s_cand = theta_hat_s_cand.detach().cpu().numpy()[0]
                        conf_s_cand = conf_param_s_cand.detach().cpu().numpy()[0]
                        conf_global_s_cand = conf_global_s_cand.detach().cpu().numpy()[0]

                        _, metrics_cand, scores_cand = generate_analysis_figure(np.clip(recon_cand, 0, 1))
                        t_cand = y_pred_s_cand[0]
                        s_cand = scores_cand["empirical_score"]
                        cnn_cand = metrics_cand["connected_components"]
                        c_cand = metrics_cand["dendrite_coverage"]

                        # 总结全局匹配度 with parameter weighting
                        # Apply weights to parameter differences
                        weighted_diff = current_weights * (y_pred_s_cand - y_pred_s)
                        H = - np.linalg.norm(weighted_diff) - (s_cand - s)

                        # save cands
                        z_cands.append(z_cand.copy())
                        H_list.append(float(H))

                        if c_cand < c or t_cand < t_val or (cnn_cand >= 3 and STRICT_UI):
                            log_container.code(f"    [Reject]c_cand={c_cand:.3f}<c={c:.3f} or t_cand={t_cand:.3f}<t={t_val:.3f} connected_components={cnn_cand}")
                            continue

                        if H > best_H_score:
                            best_H_score = H
                            best_score = s_cand
                            best_z = z_cand
                            best_img = recon_cand
                            best_params = y_pred_s_cand
                            best_params_confidence = conf_s_cand
                            best_coverage = c_cand
                            
                    except Exception as e:
                        log_container.error(f"Candidate {cand_idx} failed: {str(e)}")
                        continue

                if best_z is None:
                    log_container.code("[Stop] no valid candidate (all rejected).", language="text")
                    break
                else:
                    log_container.code(f"[Next] find best candidate with H score={best_H_score:.2f}", language="text")

                z = best_z
                s = best_score
                c = best_coverage
                t_val = best_params[0]
                y_pred_s = best_params

                # Ensure best_img is properly formatted for display
                best_img_for_display = ensure_valid_image(best_img)

                # Record hopping strength and parameter weights for this step
                st.session_state.explore_hist["hopping_strength"].append(current_hopping)
                st.session_state.explore_hist["param_weights"].append(st.session_state.param_weights.copy())

                _update_live(step, best_img_for_display, best_z, best_params, best_params_confidence, best_score, best_coverage,
                             cand_clouds=np.stack(z_cands, axis=0), cand_H=np.array(H_list, dtype=float))

                progress_bar.progress(step / STEPS_UI)

    if len(st.session_state.explore_hist["step"]) > 0:
        if len(st.session_state.explore_hist["step"]) == STEPS_UI:
            st.success(f"✅ Finished. Accepted steps: {len(st.session_state.explore_hist['z']) - 1}")
        else:
            st.warning(f"✅ Finished due to no matched candidates. Accepted steps: {len(st.session_state.explore_hist['z']) - 1}")

    # ---- History: thumbnails (default all) + playback + per-step params table ----
    hist_box = st.container(border=True)
    with hist_box:
        st.markdown("### History")

        hist = st.session_state.explore_hist
        n_hist = len(hist.get("step", []))

        if n_hist == 0:
            st.info("No history yet. Run exploration to populate.")
            st.session_state["tab5_view_step"] = 0

        else:
            st.caption(f"Total steps: {n_hist}")
            # Ensure view_step exists and is valid
            if "tab5_view_step" not in st.session_state:
                st.session_state["tab5_view_step"] = n_hist - 1
            st.session_state["tab5_view_step"] = int(np.clip(st.session_state["tab5_view_step"], 0, n_hist - 1))

            # Manual step selector (still useful when not playing)
            view_step = st.slider(
                "View step",
                min_value=0,
                max_value=n_hist - 1,
                value=int(st.session_state["tab5_view_step"]),
                key="tab5_view_step",
            )

            st.divider()

            # -------------------------
            # Thumbnails (default: ALL)
            # -------------------------
            st.markdown("#### Thumbnails")

            # Optional: layout controls
            thumb_cols = st.slider("Columns", min_value=4, max_value=12, value=8, step=1, key="tab5_thumb_cols")

            # Render all thumbnails in a grid
            idxs = list(range(n_hist))
            rows = (n_hist + thumb_cols - 1) // thumb_cols

            for r in range(rows):
                cols = st.columns(thumb_cols)
                for c in range(thumb_cols):
                    i = r * thumb_cols + c
                    if i >= n_hist:
                        break
                    with cols[c]:
                        # highlight current selected step
                        is_sel = (i == int(st.session_state["tab5_view_step"]))
                        cap = f"✅ {i}" if is_sel else f"{i}"

                        # show_coolwarm expects 2D; if you store RGB recon, use channel 0
                        img = hist["recon"][i]
                        if isinstance(img, np.ndarray) and img.ndim == 3:
                            img2d = img[..., 0]
                        else:
                            img2d = img

                        img2d = ensure_valid_image(img2d)
                        if img2d is not None:
                            show_coolwarm(img2d, cap)

            st.divider()

            # -------------------------
            # Per-step parameter list / table with weights
            # -------------------------
            st.markdown("### Parameters and Weights per step")

            # Build a table: one row per step
            max_p = max(len(np.asarray(p).reshape(-1)) for p in hist.get("params", [])) if hist.get("params") else 0

            rows = []
            weight_rows = []  # Separate rows for weights for better visualization
            
            for i in range(n_hist):
                y = np.asarray(hist["params"][i]).reshape(-1) if hist.get("params") and i < len(hist["params"]) else np.array([])
                weights = hist["param_weights"][i] if i < len(hist.get("param_weights", [])) else st.session_state.param_weights
                
                row = {
                    "step": int(hist["step"][i]) if i < len(hist["step"]) else i,
                    "score": float(hist["score"][i]) if i < len(hist.get("score", [])) else np.nan,
                    "coverage": float(hist["coverage"][i]) if i < len(hist.get("coverage", [])) else np.nan,
                    "hopping_strength": float(hist["hopping_strength"][i]) if i < len(hist.get("hopping_strength", [])) else np.nan,
                }
                weight_row = {"step": int(hist["step"][i]) if i < len(hist["step"]) else i}
                
                for k in range(max_p):
                    if k < y.size:
                        param_name = param_names[k] if k < len(param_names) else f"param_{k}"
                        row[param_name] = float(y[k])
                        weight_row[param_name] = weights.get(param_name, 1.0)
                
                rows.append(row)
                weight_rows.append(weight_row)

            # Display parameter values
            st.markdown("#### Parameter Values")
            df_params = pd.DataFrame(rows)
            st.dataframe(df_params, use_container_width=True, hide_index=True)
            
            # Display parameter weights
            st.markdown("#### Parameter Weights")
            df_weights = pd.DataFrame(weight_rows)
            st.dataframe(df_weights.style.format("{:.2f}").background_gradient(cmap="coolwarm"), use_container_width=True, hide_index=True)

    # -----------------------------
    # Display results
    # -----------------------------
    if len(st.session_state.explore_hist["step"]) > 0:
        st.subheader("🧭 Latent Space Exploration Visualizations")
        
        # PCA Visualization (existing)
        enforce_color_pca = st.checkbox("Colorize candidates by H (PCA)", value=True, key="tab5_colorize_pca")
        fig_main = _plot_latent_exploration_fig(
            z_path=st.session_state.explore_hist["z"],
            cand_clouds=st.session_state.explore_hist["cand_clouds"],
            cand_values=st.session_state.explore_hist["cand_H"],
            value_name="H",
            colorize_candidates=bool(enforce_color_pca),
        )
        st.pyplot(fig_main)
        
        # NEW: t-SNE Visualization with hopping strength encoding
        st.subheader("🌌 t-SNE Latent Space Visualization")
        st.caption("Points colored by hopping strength used at each exploration step")
        
        with st.spinner("Computing t-SNE embedding... (may take 10-30 seconds)"):
            # Get hopping strengths for all steps
            hopping_strengths = st.session_state.explore_hist.get(
                'hopping_strength', 
                [RW_SIGMA_UI] * (len(st.session_state.explore_hist["step"]) + 1)
            )
            
            # Create t-SNE visualization
            fig_tsne = _plot_latent_exploration_tsne(
                z_path=st.session_state.explore_hist["z"],
                cand_clouds=st.session_state.explore_hist["cand_clouds"],
                cand_values=st.session_state.explore_hist["cand_H"],
                value_name="H",
                colorize_candidates=True,
                hopping_strengths=hopping_strengths,
                random_state=42
            )
            st.pyplot(fig_tsne)
        
        # NEW: Parameter importance analysis section
        st.subheader("🔬 Control Parameter Importance Analysis")
        st.caption("Analysis of how control parameters influence interfacial morphology metrics")
        
        # Prepare history data for analysis
        hist = st.session_state.explore_hist
        n_hist = len(hist.get("step", []))
        
        # Build parameter history DataFrame
        max_p = max(len(np.asarray(p).reshape(-1)) for p in hist.get("params", [])) if hist.get("params") else 0
        rows = []
        for i in range(n_hist):
            y = np.asarray(hist["params"][i]).reshape(-1) if hist.get("params") and i < len(hist["params"]) else np.array([])
            row = {
                "step": int(hist["step"][i]) if i < len(hist["step"]) else i,
                "score": float(hist["score"][i]) if i < len(hist.get("score", [])) else np.nan,
                "coverage": float(hist["coverage"][i]) if i < len(hist.get("coverage", [])) else np.nan,
                "z_norm": float(np.linalg.norm(np.asarray(hist["z"][i]).reshape(-1))) if i < len(hist.get("z", [])) else np.nan,
                "t": float(y[0]) if y.size > 0 else np.nan,
                "hopping_strength": float(hist["hopping_strength"][i]) if i < len(hist.get("hopping_strength", [])) else np.nan,
            }
            for k in range(max_p):
                if k < y.size:
                    row[param_names[k]] = float(y[k])
            rows.append(row)
        
        df_history = pd.DataFrame(rows)
        
        # Perform importance analysis
        with st.spinner("Analyzing parameter importance..."):
            importance_fig, importance_df = analyze_parameter_importance(df_history)
        
        # Display results
        st.pyplot(importance_fig)
        
        # Show top influential parameters
        st.subheader("Top Influential Parameters")
        st.dataframe(
            importance_df.head(10).style.format({
                'Score_Coefficient': '{:.3f}',
                'Coverage_Correlation': '{:.3f}',
                'Score_PValue': '{:.4f}',
                'Coverage_PValue': '{:.4f}'
            }).background_gradient(subset=['Score_Coefficient', 'Coverage_Correlation'], cmap='coolwarm'),
            use_container_width=True
        )
        
        # Detailed explanation
        st.markdown("""
        **Interpretation Guide:**
        - **Score Coefficient**: Impact on dendrite intensity score (positive = increases score)
        - **Coverage Correlation**: Relationship with dendrite coverage (positive = increases coverage)
        - **P-Values**: Statistical significance (< 0.05 considered significant)
        - Parameters with large absolute values in either metric have strong influence on morphology
        """)
        
        # NEW: Download button for importance analysis
        csv = importance_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Parameter Importance Analysis",
            data=csv,
            file_name="parameter_importance_analysis.csv",
            mime="text/csv"
        )
        
        # Existing statistics plots
        st.subheader("📈 Statistics over accepted steps")
        df_curves = pd.DataFrame({
            "step": np.arange(len(st.session_state.explore_hist['z'])),
            "score": st.session_state.explore_hist['score'],
            "coverage": st.session_state.explore_hist['coverage'],
            "z_norm": np.linalg.norm(np.asarray(st.session_state.explore_hist['z']), axis=1),
            "t": [p[0] for p in st.session_state.explore_hist["params"]],
            "hopping_strength": st.session_state.explore_hist.get('hopping_strength', [RW_SIGMA_UI] * len(st.session_state.explore_hist['z']))
        }).set_index("step")
        
        # Plot with hopping strength
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_curves.index, df_curves['score'], 'b-', label='Score')
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Dendrite Score', color='b', fontsize=12)
        ax.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax.twinx()
        ax2.plot(df_curves.index, df_curves['hopping_strength'], 'r--', label='Hopping Strength')
        ax2.set_ylabel('Hopping Strength (σ)', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('Score Evolution vs Hopping Strength', fontsize=14)
        fig.tight_layout()
        st.pyplot(fig)
        
        # Existing coverage and t plots
        st.line_chart(df_curves[["coverage"]], x_label="steps", y_label="Dendrite coverage")
        st.line_chart(df_curves[["t"]], x_label="steps", y_label="t")
        st.line_chart(df_curves[["z_norm"]],  x_label="steps", y_label="Z_norm")

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
