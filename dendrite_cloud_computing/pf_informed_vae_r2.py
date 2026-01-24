import hashlib
import os
import io
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, cm
from sklearn.manifold import TSNE  # Added for t-SNE visualization
from sklearn.linear_model import LinearRegression  # Added for parameter importance analysis
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from src.modelv11 import mdn_point_and_confidence
from src.evaluate_metrics import generate_analysis_figure
from src.dataloader import inv_scale_params, smooth_scale, inv_smooth_scale, PARAM_RANGES
from src.helper import *

# ... [existing helper functions remain unchanged] ...

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
    
    colors = ['green' if corr_coverage[p] > 0 else 'red' for p in top_params_cov]
    axes[1].barh(top_params_cov, [corr_coverage[p] for p in top_params_cov], 
                color=colors, edgecolor='darkgreen')
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

# ... [existing Streamlit setup code remains unchanged] ...

with tab5:
    st.header("Heuristic Latent Space Exploration")
    
    # ... [existing initialization code remains unchanged] ...

    # -----------------------------
    # UI: exploration hyperparameters with multiple hopping strengths
    # -----------------------------
    st.subheader("Exploration Configuration")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        STEPS_UI = st.number_input("Steps", min_value=1, max_value=500, value=30, step=1)
    with c2:
        NUM_CAND_UI = st.number_input("Candidates/step", min_value=4, max_value=256, value=32, step=4)
    with c3:
        # Modified to support multiple hopping strengths
        HOPPING_MODE = st.selectbox("Hopping Mode", 
                                   ["Single Strength", "Multiple Strengths"],
                                   index=0)
        
        if HOPPING_MODE == "Single Strength":
            RW_SIGMA_UI = st.slider("Hopping Strength (σ)", 0.01, 2.0, 0.25, 0.01)
            hopping_strengths = [RW_SIGMA_UI] * (STEPS_UI + 1)
        else:
            MIN_SIGMA = st.slider("Min Strength", 0.01, 1.0, 0.05, 0.01)
            MAX_SIGMA = st.slider("Max Strength", 0.1, 2.0, 0.5, 0.01)
            # Create linear progression of hopping strengths
            hopping_strengths = np.linspace(MIN_SIGMA, MAX_SIGMA, STEPS_UI + 1)
            st.caption(f"Strengths range from {MIN_SIGMA:.2f} to {MAX_SIGMA:.2f}")
    with c4:
        STRICT_UI = st.checkbox("Strict mode", value=False)
    with c5:
        TSNE_PERPLEXITY = st.slider("t-SNE Perplexity", 5, 100, 30, 5)

    st.caption("H = -||params(z_cand) - params(z_current)|| - (score_cand - score_current). "
               "Reject if coverage decreases or t decreases.")
    
    run_btn = st.button("🚀 Run Exploration", type="primary", disabled=(seed_image is None))

    st.markdown("---")
    
    # ... [existing history initialization remains unchanged] ...

    if run_btn and seed_image is not None:
        # ... [existing initialization code] ...
        
        # Modified exploration loop to use different hopping strengths per step
        for step in range(1, STEPS_UI + 1):
            current_hopping = hopping_strengths[step] if HOPPING_MODE == "Multiple Strengths" else RW_SIGMA_UI
            
            # 生成候选 (modified to use current_hopping)
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
                # Use current hopping strength for this step
                dz = np.random.randn(*z.shape).astype(np.float32) * current_hopping
                z_cand = z + dz
                # ... [rest of candidate evaluation remains unchanged] ...
            
            # ... [rest of exploration loop remains unchanged] ...
            
            # Record hopping strength for this step
            if 'hopping_strength' not in st.session_state.explore_hist:
                st.session_state.explore_hist['hopping_strength'] = [RW_SIGMA_UI]  # Initial step
            st.session_state.explore_hist['hopping_strength'].append(current_hopping)
            
            _update_live(step, best_img, best_z, best_params, best_params_confidence, 
                        best_score, best_coverage,
                        cand_clouds=np.stack(z_cands, axis=0), cand_H=np.array(H_list, dtype=float))

    # ... [existing history display code] ...

    if len(st.session_state.explore_hist["step"]) > 0:
        st.subheader("🧭 Latent Space Exploration Visualizations")
        
        # PCA Visualization (existing)
        enforce_color = st.checkbox("Colorize candidates by H (PCA)", value=True, key="tab5_colorize_pca")
        fig_main = _plot_latent_exploration_fig(
            z_path=st.session_state.explore_hist["z"],
            cand_clouds=st.session_state.explore_hist["cand_clouds"],
            cand_values=st.session_state.explore_hist["cand_H"],
            value_name="H",
            colorize_candidates=bool(enforce_color),
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
            }).background_gradient(subset=['Score_Coefficient', 'Coverage_Correlation'], cmap='coolwarm')
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
        st.line_chart(df_curves[["z_norm"]], x_label="steps", y_label="Z_norm")

# ... [footer remains unchanged] ...
