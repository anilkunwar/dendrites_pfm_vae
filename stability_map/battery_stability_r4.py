import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(layout="wide", page_title="Stability Map", page_icon="🔋")
st.title("🔋 Interactive Stability Map for Dendrite Suppression")
st.markdown("Adjust parameter thresholds and reference design points to explore the safe operating window. Download high-resolution PDF for your LaTeX manuscript.")

# ============================================================
# 50+ COLORMAP DEFINITIONS (includes inferno, jet, turbo, rainbow)
# ============================================================
COLORMAP_OPTIONS = {
    # Perceptually Uniform Sequential
    "viridis": "viridis", "plasma": "plasma", "inferno": "inferno", "magma": "magma", "cividis": "cividis",
    # Sequential
    "Greys": "Greys", "Purples": "Purples", "Blues": "Blues", "Greens": "Greens", "Oranges": "Oranges", "Reds": "Reds",
    "YlOrBr": "YlOrBr", "YlOrRd": "YlOrRd", "OrRd": "OrRd", "PuRd": "PuRd", "RdPu": "RdPu", "BuPu": "BuPu",
    "GnBu": "GnBu", "PuBu": "PuBu", "YlGnBu": "YlGnBu", "PuBuGn": "PuBuGn", "BuGn": "BuGn", "YlGn": "YlGn",
    # Sequential (2)
    "binary": "binary", "gist_yarg": "gist_yarg", "gist_gray": "gist_gray", "gray": "gray", "bone": "bone",
    "pink": "pink", "spring": "spring", "summer": "summer", "autumn": "autumn", "winter": "winter", "cool": "cool",
    "Wistia": "Wistia", "hot": "hot", "afmhot": "afmhot", "gist_heat": "gist_heat", "copper": "copper",
    # Diverging
    "PiYG": "PiYG", "PRGn": "PRGn", "BrBG": "BrBG", "PuOr": "PuOr", "RdGy": "RdGy", "RdBu": "RdBu",
    "RdYlBu": "RdYlBu", "RdYlGn": "RdYlGn", "Spectral": "Spectral", "coolwarm": "coolwarm", "bwr": "bwr", "seismic": "seismic",
    # Qualitative
    "tab10": "tab10", "tab20": "tab20", "tab20b": "tab20b", "tab20c": "tab20c",
    "Pastel1": "Pastel1", "Pastel2": "Pastel2", "Paired": "Paired", "Accent": "Accent", "Dark2": "Dark2", "Set1": "Set1", "Set2": "Set2", "Set3": "Set3",
    # Miscellaneous
    "flag": "flag", "prism": "prism", "ocean": "ocean", "gist_earth": "gist_earth", "terrain": "terrain",
    "gist_stern": "gist_stern", "gnuplot": "gnuplot", "gnuplot2": "gnuplot2", "CMRmap": "CMRmap",
    "cubehelix": "cubehelix", "brg": "brg", "hsv": "hsv", 
    # Turbo and Rainbow (explicitly requested)
    "turbo": "turbo", "rainbow": "rainbow", "jet": "jet", "nipy_spectral": "nipy_spectral", "gist_rainbow": "gist_rainbow", "gist_ncar": "gist_ncar"
}

# ============================================================
# SIDEBAR: CONTROL PARAMETERS
# ============================================================
st.sidebar.header("⚙️ Control Parameters")

# Default values based on manuscript
default_names = [
    '$A_s$\n(J/mol)', 
    '$\\kappa$\n($10^{-10}$ J/m)', 
    '$U$\n(V)', 
    '$\\psi$\n($10^{-3}$ s$^{-1}$)'
]
default_mins = [0.0, 1.0, -0.5, 0.0]
default_maxs = [1.0, 10.0, -0.2, 5.0]
default_thresholds = [0.05, 4.0, -0.33, 1.5]
default_directions = ['lower', 'lower', 'lower', 'upper']
default_ref = [0.3, 6.0, -0.30, 1.0]

params = []
mins = []
maxs = []
thresholds = []
directions = []
ref_point = []

for i in range(4):
    # Clean display name (remove newlines for header)
    display_name = default_names[i].replace(chr(10), ' ')
    st.sidebar.subheader(f"Param {i+1}: {display_name}")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_val = st.number_input(f"Min", value=default_mins[i], key=f"min_{i}", format="%.3f")
    with col2:
        max_val = st.number_input(f"Max", value=default_maxs[i], key=f"max_{i}", format="%.3f")

    if min_val >= max_val:
        st.sidebar.error("Max must be > Min!")
        max_val = min_val + 0.1

    thresh = st.number_input(f"Threshold", value=default_thresholds[i], key=f"thresh_{i}", format="%.3f")
    dir_val = st.selectbox(
        f"Safe Region is", 
        options=['> Threshold (Lower bound)', '< Threshold (Upper bound)'], 
        index=0 if default_directions[i] == 'lower' else 1, 
        key=f"dir_{i}"
    )

    ref = st.number_input(f"Reference Point", value=default_ref[i], key=f"ref_{i}", format="%.3f")

    params.append(default_names[i])
    mins.append(min_val)
    maxs.append(max_val)
    thresholds.append(thresh)
    directions.append('lower' if 'Lower' in dir_val else 'upper')
    ref_point.append(ref)

# ============================================================
# SIDEBAR: VISUAL ENHANCEMENT CONTROLS
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("🎨 Visual Enhancements")

# Colormap selection
colormap_choice = st.sidebar.selectbox(
    "Colormap for Safe Zones",
    options=list(COLORMAP_OPTIONS.keys()),
    index=list(COLORMAP_OPTIONS.keys()).index("inferno"),
    help="Choose from 50+ colormaps including inferno, jet, turbo, rainbow"
)

# Font size controls
st.sidebar.subheader("Font Sizes")
label_fontsize = st.sidebar.slider("Parameter Label Size", 8, 24, 12, help="Font size for parameter names at top")
tick_fontsize = st.sidebar.slider("Tick Label Size", 6, 20, 10, help="Font size for axis tick values")
threshold_fontsize = st.sidebar.slider("Threshold Label Size", 6, 20, 10, help="Font size for threshold values")
ref_fontsize = st.sidebar.slider("Reference Point Size", 6, 20, 10, help="Font size for reference point labels")
title_fontsize = st.sidebar.slider("Title Size", 10, 30, 14, help="Font size for plot title")

# Line and marker controls
st.sidebar.subheader("Line & Marker Styles")
line_width = st.sidebar.slider("Reference Line Width", 0.5, 5.0, 2.5, 0.1)
marker_size = st.sidebar.slider("Marker Size", 4, 20, 8, 1)
threshold_linewidth = st.sidebar.slider("Threshold Line Width", 0.5, 4.0, 1.5, 0.1)

# Alpha/Transparency controls
st.sidebar.subheader("Transparency")
safe_alpha = st.sidebar.slider("Safe Zone Alpha", 0.1, 1.0, 0.3, 0.05)
unsafe_alpha = st.sidebar.slider("Unsafe Zone Alpha", 0.1, 1.0, 0.2, 0.05)

# Color customization
st.sidebar.subheader("Custom Colors")
safe_color = st.sidebar.color_picker("Safe Zone Color", "#00FF00", help="Override colormap with solid color")
unsafe_color = st.sidebar.color_picker("Unsafe Zone Color", "#FF0000", help="Override colormap with solid color")
use_custom_colors = st.sidebar.checkbox("Use Custom Colors Instead of Colormap", value=False)
ref_line_color = st.sidebar.color_picker("Reference Line Color", "#0000FF")
threshold_line_color = st.sidebar.color_picker("Threshold Line Color", "#FF0000")

# Figure dimensions
st.sidebar.subheader("Figure Dimensions")
fig_width = st.sidebar.slider("Figure Width (inches)", 6, 20, 10, 1)
fig_height = st.sidebar.slider("Figure Height (inches)", 4, 16, 6, 1)

# Grid and background
st.sidebar.subheader("Grid & Background")
show_grid = st.sidebar.checkbox("Show Grid Lines", value=False)
grid_alpha = st.sidebar.slider("Grid Alpha", 0.1, 1.0, 0.3, 0.05)
background_color = st.sidebar.color_picker("Background Color", "#FFFFFF")

# ============================================================
# PLOTTING FUNCTION
# ============================================================
def create_plot(params, mins, maxs, thresholds, directions, ref_point):
    # Normalize values
    ref_norm = [(ref_point[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(params))]
    thresh_norm = [(thresholds[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(params))]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    N = len(params)
    x_pos = np.arange(1, N + 1)  # Integer labels: 1, 2, 3, 4...

    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(0.5, N + 0.5)  # Adjusted for 1-based indexing
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Grid
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle='--', axis='y')

    # X-axis: integer denominations for control variable numbers
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in x_pos], fontsize=tick_fontsize, fontweight='bold')
    ax.set_xlabel("Control Variable Number", fontsize=label_fontsize, fontweight='bold')

    # Get colormap
    cmap = plt.cm.get_cmap(COLORMAP_OPTIONS[colormap_choice])

    for i in range(N):
        # Main vertical axis line
        ax.plot([x_pos[i], x_pos[i]], [0, 1], color='black', linewidth=1.5, zorder=1)

        # Parameter label at top (with LaTeX support)
        ax.text(x_pos[i], 1.05, params[i], ha='center', va='bottom', 
                fontsize=label_fontsize, fontweight='bold')

        # Tick marks and values
        ticks = np.linspace(0, 1, 5)
        tick_vals = np.linspace(mins[i], maxs[i], 5)
        for t, val in zip(ticks, tick_vals):
            ax.plot([x_pos[i]-0.02, x_pos[i]+0.02], [t, t], color='black', linewidth=1)
            ax.text(x_pos[i]-0.08, t, f'{val:.2f}', ha='right', va='center', 
                    fontsize=tick_fontsize)

        # Clip normalized values to [0, 1]
        t_norm = max(0, min(1, thresh_norm[i]))

        # Determine colors
        if use_custom_colors:
            safe_c = safe_color
            unsafe_c = unsafe_color
        else:
            # Use colormap: map normalized threshold to colormap
            safe_c = cmap(0.7)  # Upper range of colormap
            unsafe_c = cmap(0.2)  # Lower range of colormap

        # Fill safe/unsafe regions
        if directions[i] == 'lower':
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], t_norm, 1, 
                           color=safe_c, alpha=safe_alpha, zorder=0)
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], 0, t_norm, 
                           color=unsafe_c, alpha=unsafe_alpha, zorder=0)
            ax.plot([x_pos[i]-0.08, x_pos[i]+0.08], [t_norm, t_norm], 
                   color=threshold_line_color, linestyle='--', 
                   linewidth=threshold_linewidth, zorder=2)
            ax.text(x_pos[i]+0.12, t_norm, f'{thresholds[i]:.3f}', 
                   va='center', ha='left', color=threshold_line_color, 
                   fontsize=threshold_fontsize, fontweight='bold')
        else:
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], 0, t_norm, 
                           color=safe_c, alpha=safe_alpha, zorder=0)
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], t_norm, 1, 
                           color=unsafe_c, alpha=unsafe_alpha, zorder=0)
            ax.plot([x_pos[i]-0.08, x_pos[i]+0.08], [t_norm, t_norm], 
                   color=threshold_line_color, linestyle='--', 
                   linewidth=threshold_linewidth, zorder=2)
            ax.text(x_pos[i]+0.12, t_norm, f'{thresholds[i]:.3f}', 
                   va='center', ha='left', color=threshold_line_color, 
                   fontsize=threshold_fontsize, fontweight='bold')

    # Reference point line
    r_norm = [max(0, min(1, rn)) for rn in ref_norm]
    ax.plot(x_pos, r_norm, color=ref_line_color, linewidth=line_width, 
            zorder=3, marker='o', markersize=marker_size, 
            markerfacecolor=ref_line_color, markeredgecolor='white', 
            markeredgewidth=1.5, label='Reference Design')

    for i in range(N):
        ax.text(x_pos[i], r_norm[i] + 0.03, f'{ref_point[i]:.3f}', 
               ha='center', va='bottom', fontsize=ref_fontsize, 
               color=ref_line_color, fontweight='bold')

    # Title
    plt.title('Stability Map for Dendrite Suppression', 
             fontsize=title_fontsize, pad=20, fontweight='bold')

    # Legend
    ax.legend(loc='upper right', fontsize=tick_fontsize, 
             framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    return fig

# ============================================================
# GENERATE AND DISPLAY PLOT
# ============================================================
fig = create_plot(params, mins, maxs, thresholds, directions, ref_point)
st.pyplot(fig)

# ============================================================
# DOWNLOAD BUTTONS (PDF + PNG + SVG)
# ============================================================
st.markdown("---")
st.subheader("📥 Export Options")

col1, col2, col3 = st.columns(3)

with col1:
    # PDF Export
    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches='tight', dpi=300)
    st.download_button(
       label="📄 Download PDF (LaTeX)",
       data=buf_pdf.getvalue(),
       file_name="stability_map.pdf",
       mime="application/pdf"
    )

with col2:
    # PNG Export
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", bbox_inches='tight', dpi=300)
    st.download_button(
       label="🖼️ Download PNG (300 DPI)",
       data=buf_png.getvalue(),
       file_name="stability_map.png",
       mime="image/png"
    )

with col3:
    # SVG Export
    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format="svg", bbox_inches='tight')
    st.download_button(
       label="🎨 Download SVG (Vector)",
       data=buf_svg.getvalue(),
       file_name="stability_map.svg",
       mime="image/svg+xml"
    )

# ============================================================
# SUMMARY STATISTICS
# ============================================================
st.markdown("---")
st.subheader("📊 Parameter Summary")

summary_data = []
for i in range(4):
    status = "✅ SAFE" if (
        (directions[i] == 'lower' and ref_point[i] >= thresholds[i]) or 
        (directions[i] == 'upper' and ref_point[i] <= thresholds[i])
    ) else "❌ UNSAFE"

    summary_data.append({
        "Variable": f"Param {i+1}",
        "Name": params[i].replace(chr(10), ' '),
        "Min": f"{mins[i]:.3f}",
        "Max": f"{maxs[i]:.3f}",
        "Threshold": f"{thresholds[i]:.3f}",
        "Reference": f"{ref_point[i]:.3f}",
        "Direction": directions[i].upper(),
        "Status": status
    })

st.table(summary_data)

# Safety score
safe_count = sum(1 for i in range(4) if 
    (directions[i] == 'lower' and ref_point[i] >= thresholds[i]) or 
    (directions[i] == 'upper' and ref_point[i] <= thresholds[i]))

st.metric("Safety Score", f"{safe_count}/4 parameters in safe zone", 
         delta=f"{safe_count - 2} vs baseline" if safe_count != 2 else "Baseline")
