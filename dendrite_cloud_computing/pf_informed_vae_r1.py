import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors, cm

from src.evaluate_metrics import generate_analysis_figure
from src.dataloader import inv_scale_params, smooth_scale, inv_smooth_scale, PARAM_RANGES
from src.modelv11 import mdn_point_and_confidence
from src.helper import *

def show_coolwarm(gray_image, caption, container=None):
    norm = colors.Normalize(vmin=gray_image.min(), vmax=gray_image.max())
    colored_img = cm.coolwarm(norm(gray_image))  # shape: (H, W, 4)
    if container is None:
        st.image(colored_img, caption=caption, width='stretch')
    else:
        container.image(colored_img, caption=caption, width='stretch')

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
device = "cpu"
model = load_model(device)
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
        NUM_CAND_UI = st.number_input("Number of candidates each step", min_value=4, max_value=256, value=48, step=4, key="tab5_num_cand")
    with c3:
        RW_SIGMA_UI = st.slider("RW_SIGMA", 0.01, 2.0, 0.25, 0.01, key="tab5_sigma")
    with c4:
        enforce_color = st.checkbox("Colorize candidates by H", value=True, key="tab5_colorize")

    st.caption("H = -||params(z_cand) - params(z_current)|| - (score_cand - score_current). "
               "Reject if coverage decreases or t decreases.")

    run_btn = st.button("🚀 Run Exploration", type="primary", disabled=(seed_image is None))

    st.markdown("---")

    # -----------------------------
    # TAB5: Live viewer + history + params/score panel
    # -----------------------------
    st.subheader("🖼️ Live Exploration Viewer")

    # --- Session state (history) ---
    if "explore_hist" not in st.session_state:
        st.session_state.explore_hist = {
            "recon": [],  # list of (H,W,3) float
            "analysis_fig": [],  # list of matplotlib figs (optional)
            "z": [],  # list of (D,)
            "params": [],  # list of (P,) y_pred_s
            "params_confidence": [],
            "score": [],  # list of float
            "coverage": [],  # list of float
            "step": [],  # list of int
        }
    if "log_lines" not in st.session_state:
        st.session_state.log_lines = []

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
                     cand_H_list: np.ndarray | None = None):
        """
        Append history + refresh the Live viewer + refresh metrics table + refresh candidate summary.
        Safe for repeated calls.
        """
        hist = st.session_state.explore_hist

        hist["step"].append(int(step_i))
        hist["recon"].append(np.asarray(recon_rgb))
        hist["z"].append(np.asarray(z_1d).reshape(-1))
        hist["params"].append(np.asarray(y_pred_s).reshape(-1))
        hist["params_confidence"].append(np.asarray(y_pred_conf).reshape(-1))
        hist["score"].append(float(score))
        hist["coverage"].append(float(coverage))

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
        metrics_placeholder.dataframe(df, width='stretch', hide_index=True)

    # show results in one step
    log_container = st.container(height=220)
    with log_container:
        st.code(
            "\n".join(st.session_state.log_lines[-300:]),  # 防止无限增长
            language="text"
        )

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

        # show selected historical by default (updates after run)
        if len(st.session_state.explore_hist["step"]) > 0:
            i = int(st.session_state.get("tab5_view_step", len(st.session_state.explore_hist["step"]) - 1))
            show_coolwarm(st.session_state.explore_hist["recon"][i], caption=f"Step {i}")
            live_caption_placeholder.caption("Tip: during a run, this panel updates every accepted step.")

    # ---- Metrics / params panel for selected step ----
    with metrics_box:
        st.markdown("### Params & Scores")

        metrics_placeholder = st.empty()

        if len(st.session_state.explore_hist["step"]) > 0:
            i = int(st.session_state.get("tab5_view_step", len(st.session_state.explore_hist["step"]) - 1))
            y_pred = st.session_state.explore_hist["params"][i]
            y_confidence = st.session_state.explore_hist["params_confidence"][i]
            extra = {
                "score": st.session_state.explore_hist["score"][i],
                "coverage": st.session_state.explore_hist["coverage"][i],
                "||z||": float(np.linalg.norm(st.session_state.explore_hist["z"][i])),
            }
            df = _params_to_table(y_pred, y_confidence, extra)
            metrics_placeholder.dataframe(df, width='stretch', hide_index=True)
        else:
            metrics_placeholder.info("Metrics table will appear after the first accepted step.")

    # ---- Controls: history browsing ----
    hist_box = st.container(border=True)
    with hist_box:
        st.markdown("### History")
        hist = st.session_state.explore_hist
        n_hist = len(hist["step"])
        if n_hist == 0:
            st.info("No history yet. Run exploration to populate.")
            view_step = 0
        else:
            view_step = st.slider("View step", min_value=0, max_value=n_hist, value=n_hist,
                                  key="tab5_view_step")

            # Compact thumbnail strip (optional)
            show_thumbs = st.checkbox("Show thumbnails", value=False, key="tab5_show_thumbs")
            if show_thumbs:
                # show up to 10 evenly spaced thumbs
                idxs = np.linspace(0, n_hist - 1, num=min(10, n_hist), dtype=int).tolist()
                cols = st.columns(len(idxs))
                for col, i in zip(cols, idxs):
                    with col:
                        show_coolwarm(hist["recon"][i], f"{i}")

    if run_btn and seed_image is not None:

        with st.spinner("Running exploration..."):

            progress_bar = st.progress(0)
            with torch.no_grad():
                seed_image_tensor = prep_tensor_from_image(seed_image, expected_size)
                recon, mu_q, logvar_q, (pi_s, mu_s, log_sigma_s), z = model(seed_image_tensor)
                # ---- stochastic prediction + confidence (from sampled z) ----
                theta_hat_s, conf_param_s, conf_global_s, modes_s = mdn_point_and_confidence(
                    pi_s, mu_s, log_sigma_s, var_scale=var_scale
                )

            recon = inv_smooth_scale(recon)
            recon = recon.cpu().detach().numpy()[0, 0]
            z = z.cpu().detach().numpy()[0]
            y_pred_s = theta_hat_s.detach().cpu().numpy()[0]
            conf_s = conf_param_s.detach().cpu().numpy()[0]
            conf_global_s = conf_global_s.detach().cpu().numpy()[0]

            _, metrics, scores = generate_analysis_figure(np.clip(recon, 0, 1))
            s = scores["empirical_score"]
            c = metrics["dendrite_coverage"]
            t = y_pred_s[0]
            _update_live(0, recon, z, y_pred_s, conf_s, s, c)

            z_path = [z.copy()]
            cand_clouds = []
            cand_H = []
            score_path = [float(s)]
            coverage_path = [float(c)]
            for step in range(1, STEPS_UI + 1):
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
                for _ in range(NUM_CAND_UI):

                    dz = np.random.randn(*z.shape).astype(np.float32) * RW_SIGMA_UI
                    z_cand = z + dz
                    z_cand_tensor = torch.from_numpy(z_cand).unsqueeze(0).to(device)
                    with torch.no_grad():
                        recon_cand, (theta_hat_s_cand, conf_param_s_cand, conf_global_s_cand, modes_s_cand) = \
                            inference(model, z_cand_tensor, var_scale=var_scale)
                    recon_cand = inv_smooth_scale(recon_cand)
                    recon_cand = recon_cand.cpu().detach().numpy()[0, 0]
                    y_pred_s_cand = theta_hat_s_cand.detach().cpu().numpy()[0]
                    conf_s_cand = conf_param_s_cand.detach().cpu().numpy()[0]
                    conf_global_s_cand = conf_global_s_cand.detach().cpu().numpy()[0]

                    _, metrics_cand, scores_cand = generate_analysis_figure(np.clip(recon_cand, 0, 1))
                    t_cand = y_pred_s_cand[0]
                    s_cand = scores_cand["empirical_score"]
                    c_cand = metrics_cand["dendrite_coverage"]

                    # 总结全局匹配度
                    H = - np.linalg.norm(y_pred_s_cand - y_pred_s) - (s_cand - s)

                    # save cands
                    z_cands.append(z_cand.copy())
                    H_list.append(float(H))

                    if c_cand < c or t_cand < t:
                        st.session_state.log_lines.append(f"    [Reject]c_cand={c_cand:.3f}<c={c:.3f} or t_cand={t_cand:.3f}<t={t:.3f}")
                        continue

                    if H > best_H_score:
                        best_H_score = H
                        best_score = s_cand
                        best_z = z_cand
                        best_img = recon_cand
                        best_params = y_pred_s_cand
                        best_params_confidence = conf_s_cand
                        best_coverage = c_cand

                if best_z is None:
                    st.session_state.log_lines.append("[Stop] no valid candidate (all rejected).")
                    break
                else:
                    st.session_state.log_lines.append(f"[Next] find best candidate with H score={best_H_score:.2f}")

                z = best_z
                s = best_score
                c = best_coverage
                t = best_params[0]
                y_pred_s = best_params

                _update_live(0, best_img, best_z, best_params, best_params_confidence, best_score, best_coverage, cand_H)

                z_path.append(z.copy())
                score_path.append(float(s))
                coverage_path.append(float(c))

                cand_clouds.append(np.stack(z_cands, axis=0))  # (NUM_CAND, D)
                cand_H.append(np.array(H_list, dtype=float))  # (NUM_CAND,)

                progress_bar.progress(step / STEPS_UI)

        # -----------------------------
        # Display results
        # -----------------------------
        st.success(f"✅ Finished. Accepted steps: {len(z_path) - 1}")

        # st.subheader("🧭 Latent exploration visualization")
        # fig_main, fig_norm = _plot_latent_exploration_fig(
        #     z_path=z_path,
        #     cand_clouds=cand_clouds,
        #     cand_values=cand_H,
        #     value_name="H",
        #     colorize_candidates=bool(enforce_color),
        # )
        # st.pyplot(fig_main)
        # st.pyplot(fig_norm)
        #
        # # score / coverage curves
        # st.subheader("📈 Score / Coverage over accepted steps")
        # df_curves = pd.DataFrame({
        #     "step": np.arange(len(score_path)),
        #     "score": score_path,
        #     "coverage": coverage_path,
        #     "z_norm": np.linalg.norm(np.asarray(z_path), axis=1),
        # }).set_index("step")
        # st.line_chart(df_curves[["score", "coverage", "z_norm"]])

        # # show a compact gallery: first, last, and a few intermediates
        # st.subheader("🖼️ Reconstructions along the accepted path")
        # show_idx = list(dict.fromkeys(
        #     [0, min(1, len(recon_path) - 1), len(recon_path) // 4, len(recon_path) // 2, (3 * len(recon_path)) // 4,
        #      len(recon_path) - 1]
        # ))
        # cols = st.columns(len(show_idx))
        # for col, idx in zip(cols, show_idx):
        #     with col:
        #         st.markdown(f"**step {idx}**")
        #         show_coolwarm(recon_path[idx][..., 0], caption=f"recon step {idx}")
        #
        # st.subheader("🔍 Detailed analysis figures (optional)")
        # if st.checkbox("Show analysis figure for each accepted step (can be slow)", value=False, key="tab5_show_all"):
        #     for i, fig in enumerate(analysis_figs):
        #         with st.container(border=True):
        #             st.markdown(f"**Accepted step {i}**  · score={score_path[i]:.4f} · coverage={coverage_path[i]:.4f}")
        #             st.pyplot(fig)

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
