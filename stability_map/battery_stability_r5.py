import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io

# --- 1. Journal-Quality Configuration ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight'
})

# --- 2. Page Configuration ---
st.set_page_config(layout="wide")
st.title("🔋 Interactive Stability Map for Dendrite Suppression")
st.markdown("""
Adjust the parameter thresholds and reference design points below to explore the safe operating window. 
You can download the high-resolution figures directly for your LaTeX manuscript.
""")

# --- 3. Sidebar: Appearance & Style ---
st.sidebar.header("🎨 Figure Appearance")

# Font Controls
title_font = st.sidebar.slider("Title Font Size", 10, 40, 20)
label_font = st.sidebar.slider("Parameter Label Font Size", 8, 30, 16)
tick_font = st.sidebar.slider("Tick Font Size", 6, 24, 12)
threshold_font = st.sidebar.slider("Threshold Font Size", 6, 24, 12)
reference_font = st.sidebar.slider("Reference Font Size", 6, 24, 12)

# Dimension Controls
fig_width = st.sidebar.slider("Figure Width (inches)", 6, 20, 12)
fig_height = st.sidebar.slider("Figure Height (inches)", 4, 12, 7)

# Line and Marker Controls
line_width = st.sidebar.slider("Axis Line Width", 0.5, 5.0, 2.0)
trajectory_width = st.sidebar.slider("Trajectory Line Width", 1.0, 8.0, 3.0)
marker_size = st.sidebar.slider("Marker Size", 2, 20, 8)
dpi = st.sidebar.slider("Export DPI", 300, 1200, 600)

# === BAR THICKNESS SLIDER (RESTORED ORIGINAL BEHAVIOR) ===
bar_thickness = st.sidebar.slider("Bar Thickness", 0.02, 0.30, 0.08, step=0.01,
    help="Width of the safe/unsafe color bands (original default was 0.08)")

# Style Toggles
threshold_style = st.sidebar.selectbox("Threshold Line Style", ["--", "-", "-.", ":"])
show_grid = st.sidebar.checkbox("Show Grid", True)
dark_theme = st.sidebar.checkbox("Dark Theme", False)
show_colorbar = st.sidebar.checkbox("Show Colorbar", True)

# === LEGEND CONTROLS ===
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Legend Controls")
show_legend = st.sidebar.checkbox("Show Legend", True)

legend_locations = [
    'best', 'upper right', 'upper left', 'lower left', 'lower right',
    'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
]
legend_loc = st.sidebar.selectbox("Legend Location", legend_locations, index=0)
legend_ncol = st.sidebar.slider("Legend Columns", 1, 3, 1)
legend_frame = st.sidebar.checkbox("Legend Frame", True)
legend_alpha = st.sidebar.slider("Legend Frame Alpha", 0.0, 1.0, 0.9)

# === LABEL PADDING & OFFSET CONTROLS ===
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Label Padding & Offsets")

# Parameter label (top) vertical padding
param_label_pad = st.sidebar.slider("Param Label Vertical Pad", 0.0, 0.15, 0.05, step=0.01,
    help="Vertical distance between parameter name and top of axis bar")

# Tick label horizontal padding
tick_label_pad = st.sidebar.slider("Tick Label Horizontal Pad", 0.02, 0.20, 0.08, step=0.01,
    help="Horizontal distance between tick marks and their numeric labels")

# Threshold label horizontal padding
threshold_label_pad = st.sidebar.slider("Threshold Label Horizontal Pad", 0.05, 0.30, 0.12, step=0.01,
    help="Horizontal distance between threshold line and its value label")

# Reference point label vertical padding
ref_label_pad = st.sidebar.slider("Reference Label Vertical Pad", 0.01, 0.10, 0.03, step=0.005,
    help="Vertical distance between reference marker and its value label")

# Reference label offset direction
ref_label_position = st.sidebar.selectbox(
    "Reference Label Position",
    ["Above", "Below", "Auto (smart)"],
    index=2,
    help="'Auto' places label above if space permits, otherwise below to avoid overlap"
)

# Minimum vertical spacing between labels
min_label_spacing = st.sidebar.slider("Min Label Spacing", 0.0, 0.10, 0.02, step=0.005,
    help="Minimum vertical gap to prevent threshold and reference labels from overlapping")

# Colormap Selection
st.sidebar.markdown("---")
available_colormaps = sorted([
    'viridis','plasma','inferno','magma','cividis',
    'turbo','jet','rainbow','nipy_spectral','gist_rainbow',
    'cool','hot','spring','summer','autumn','winter',
    'Wistia','afmhot','copper','Spectral','RdYlGn',
    'RdBu','PiYG','PRGn','BrBG','PuOr',
    'coolwarm','bwr','seismic',
    'twilight','twilight_shifted','hsv',
    'Pastel1','Pastel2','Paired','Accent',
    'Dark2','Set1','Set2','Set3',
    'tab10','tab20','tab20b','tab20c',
    'flag','prism','ocean','terrain',
    'gist_earth','cubehelix','gnuplot',
    'gnuplot2','CMRmap','gist_stern',
    'bone','pink','gray','Greys'
])

selected_cmap = st.sidebar.selectbox(
    "Colormap",
    available_colormaps,
    index=0
)

# --- 4. Sidebar: Control Parameters ---
st.sidebar.header("⚙️ Control Parameters")

default_names = [r'$A_s$' + '\n(J/mol)', r'$\kappa$' + '\n($10^{-10}$ J/m)', r'$U$' + '\n(V)', r'$\psi$' + '\n($10^{-3}$ s$^{-1}$)']
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

for i in range(len(default_names)):
    st.sidebar.subheader(f"Param {i+1}: {default_names[i].replace(chr(10), ' ')}")
    
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

# --- 5. Plotting Function ---
def create_plot(params, mins, maxs, thresholds, directions, ref_point, 
                title_font, label_font, tick_font, threshold_font, reference_font,
                fig_width, fig_height, line_width, trajectory_width, marker_size, 
                show_grid, dark_theme, show_legend, legend_loc, legend_ncol, legend_frame, legend_alpha,
                show_colorbar, threshold_style, selected_cmap,
                bar_thickness, param_label_pad, tick_label_pad, threshold_label_pad, ref_label_pad,
                ref_label_position, min_label_spacing):
    
    # Apply Theme
    if dark_theme:
        plt.style.use('dark_background')
        text_color = 'white'
    else:
        plt.style.use('default')
        text_color = 'black'

    # Colormap Setup
    cmap = plt.get_cmap(selected_cmap)
    safe_color = cmap(0.85) 
    unsafe_color = cmap(0.15)
    reference_color = cmap(0.55)

    # Data Normalization
    ref_norm = [(ref_point[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(params))]
    thresh_norm = [(thresholds[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(params))]

    # Initialize Figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # X-Axis Setup
    N = len(params)
    x_pos = np.arange(N)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [str(i+1) for i in range(N)],
        fontsize=tick_font,
        fontweight='bold'
    )
    ax.set_xlabel(
        "Control Variable Number",
        fontsize=label_font,
        fontweight='bold'
    )
    ax.set_xlim(-0.5, N - 0.5)

    # Y-Axis Setup
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(line_width)

    if show_grid:
        ax.grid(True, linestyle=':', alpha=0.6, zorder=0)

    # === PLOT PARAMETERS BARS (RESTORED ORIGINAL STRUCTURE) ===
    for i in range(N):
        # Main vertical axis line
        ax.plot([x_pos[i], x_pos[i]], [0, 1], color=text_color, linewidth=line_width, zorder=1)
        
        # Parameter Label (Top) — with configurable padding
        ax.text(x_pos[i], 1.0 + param_label_pad, params[i], 
                ha='center', va='bottom', fontsize=label_font, fontweight='bold')
        
        # Ticks on the bar — with configurable horizontal padding
        ticks = np.linspace(0, 1, 5)
        tick_vals = np.linspace(mins[i], maxs[i], 5)
        for t, val in zip(ticks, tick_vals):
            ax.plot([x_pos[i]-0.02, x_pos[i]+0.02], [t, t], color=text_color, linewidth=1)
            ax.text(x_pos[i] - tick_label_pad, t, f'{val:.2g}', 
                    ha='right', va='center', fontsize=tick_font)
            
        # Threshold and Safe/Unsafe Regions — USING bar_thickness SLIDER
        t_norm = max(0, min(1, thresh_norm[i]))
        
        if directions[i] == 'lower':  # Safe is > Threshold (Top region)
            ax.fill_between([x_pos[i]-bar_thickness, x_pos[i]+bar_thickness], t_norm, 1, 
                          color=safe_color, alpha=0.4, zorder=0, 
                          label='Safe Region' if i==0 else "")
            ax.fill_between([x_pos[i]-bar_thickness, x_pos[i]+bar_thickness], 0, t_norm, 
                          color=unsafe_color, alpha=0.2, zorder=0, 
                          label='Unsafe Region' if i==0 else "")
            ax.plot([x_pos[i]-bar_thickness, x_pos[i]+bar_thickness], [t_norm, t_norm], 
                   color=unsafe_color, linestyle=threshold_style, linewidth=line_width, zorder=2)
            
            # Threshold label with configurable horizontal padding — HIGH ZORDER=10
            ax.text(x_pos[i] + bar_thickness + threshold_label_pad, t_norm, 
                   f'{thresholds[i]:.2g}', va='center', ha='left', 
                   color=unsafe_color, fontsize=threshold_font, fontweight='bold', zorder=10)
        else:  # Safe is < Threshold (Bottom region)
            ax.fill_between([x_pos[i]-bar_thickness, x_pos[i]+bar_thickness], 0, t_norm, 
                          color=safe_color, alpha=0.4, zorder=0, 
                          label='Safe Region' if i==0 else "")
            ax.fill_between([x_pos[i]-bar_thickness, x_pos[i]+bar_thickness], t_norm, 1, 
                          color=unsafe_color, alpha=0.2, zorder=0, 
                          label='Unsafe Region' if i==0 else "")
            ax.plot([x_pos[i]-bar_thickness, x_pos[i]+bar_thickness], [t_norm, t_norm], 
                   color=unsafe_color, linestyle=threshold_style, linewidth=line_width, zorder=2)
            
            # Threshold label with configurable horizontal padding — HIGH ZORDER=10
            ax.text(x_pos[i] + bar_thickness + threshold_label_pad, t_norm, 
                   f'{thresholds[i]:.2g}', va='center', ha='left', 
                   color=unsafe_color, fontsize=threshold_font, fontweight='bold', zorder=10)

    # Reference Trajectory
    r_norm = [max(0, min(1, rn)) for rn in ref_norm]
    
    # Draw Line connecting reference points
    ax.plot(
        x_pos, r_norm, 
        color=reference_color, 
        linewidth=trajectory_width, 
        marker='o', 
        markersize=marker_size, 
        markeredgecolor=text_color, 
        markeredgewidth=1.0, 
        zorder=5,
        label='Reference Design'
    )
    
    # === REFERENCE VALUE LABELS — HIGH ZORDER=10 + SMART POSITIONING ===
    for i in range(N):
        t_norm = max(0, min(1, thresh_norm[i]))
        ref_val = r_norm[i]
        
        # Determine vertical position based on settings and proximity to threshold
        distance_to_thresh = abs(ref_val - t_norm)
        
        if ref_label_position == "Above":
            label_y = ref_val + ref_label_pad
            va = 'bottom'
        elif ref_label_position == "Below":
            label_y = ref_val - ref_label_pad
            va = 'top'
        else:  # Auto (smart)
            if distance_to_thresh < min_label_spacing + ref_label_pad:
                if ref_val > t_norm:
                    label_y = ref_val + ref_label_pad
                    va = 'bottom'
                else:
                    label_y = ref_val - ref_label_pad
                    va = 'top'
            elif ref_val > 0.9:
                label_y = ref_val - ref_label_pad
                va = 'top'
            elif ref_val < 0.1:
                label_y = ref_val + ref_label_pad
                va = 'bottom'
            else:
                label_y = ref_val + ref_label_pad
                va = 'bottom'
        
        # Draw reference label ON TOP with zorder=10
        ax.text(x_pos[i], label_y, f'{ref_point[i]:.2g}', 
               ha='center', va=va, 
               fontsize=reference_font, color=reference_color, fontweight='bold', zorder=10)

    # Final Styling
    ax.set_title("Stability Map for Dendrite Suppression", 
                fontsize=title_font, fontweight='bold', pad=25)
    
    # Legend
    if show_legend:
        legend = ax.legend(
            loc=legend_loc,
            ncol=legend_ncol,
            frameon=legend_frame,
            framealpha=legend_alpha,
            fontsize=tick_font
        )
        for text in legend.get_texts():
            text.set_color(text_color)

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Normalized Stability Scale", fontsize=label_font)
        cbar.ax.tick_params(labelsize=tick_font)

    plt.tight_layout()
    return fig

# --- 6. Main Execution ---
fig = create_plot(
    params, mins, maxs, thresholds, directions, ref_point, 
    title_font, label_font, tick_font, threshold_font, reference_font,
    fig_width, fig_height, line_width, trajectory_width, marker_size,
    show_grid, dark_theme, show_legend, legend_loc, legend_ncol, legend_frame, legend_alpha,
    show_colorbar, threshold_style, selected_cmap,
    bar_thickness, param_label_pad, tick_label_pad, threshold_label_pad, ref_label_pad,
    ref_label_position, min_label_spacing
)

st.pyplot(fig)

# --- 7. Export Buttons ---
st.header("📥 Download Figures")
col1, col2, col3 = st.columns(3)

with col1:
    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format='pdf', dpi=dpi)
    st.download_button(
        label="Download PDF",
        data=pdf_buf.getvalue(),
        file_name="stability_map.pdf",
        mime="application/pdf"
    )

with col2:
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format='png', dpi=dpi, transparent=True)
    st.download_button(
        label="Download PNG",
        data=png_buf.getvalue(),
        file_name="stability_map.png",
        mime="image/png"
    )

with col3:
    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format='svg')
    st.download_button(
        label="Download SVG",
        data=svg_buf.getvalue(),
        file_name="stability_map.svg",
        mime="image/svg+xml"
    )
