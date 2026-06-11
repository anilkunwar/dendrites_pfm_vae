import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io

# 20. Automatic journal-quality rcParams
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight'
})

st.set_page_config(layout="wide", page_title="Stability Map Generator")
st.title("🔋 Interactive Stability Map for Dendrite Suppression")
st.markdown("Adjust the parameters, styling, and export options to generate a publication-quality parallel coordinates plot.")

# ==========================================
# 1-17. Sidebar: Figure Appearance & Controls
# ==========================================
st.sidebar.header("Figure Appearance")

title_font = st.sidebar.slider("Title Font Size", 10, 40, 20)
label_font = st.sidebar.slider("Parameter Label Font Size", 8, 30, 16)
tick_font = st.sidebar.slider("Tick Font Size", 6, 24, 12)
threshold_font = st.sidebar.slider("Threshold Font Size", 6, 24, 12)
reference_font = st.sidebar.slider("Reference Font Size", 6, 24, 12)

fig_width = st.sidebar.slider("Figure Width (inches)", 6, 20, 12)
fig_height = st.sidebar.slider("Figure Height (inches)", 4, 12, 7)

line_width = st.sidebar.slider("Axis Line Width", 0.5, 5.0, 2.0)
trajectory_width = st.sidebar.slider("Trajectory Width", 1.0, 8.0, 3.0)

marker_size = st.sidebar.slider("Marker Size", 2, 20, 8)

dpi = st.sidebar.slider("Export DPI", 300, 1200, 600)

show_grid = st.sidebar.checkbox("Show Grid", True)
dark_theme = st.sidebar.checkbox("Dark Theme", False)
transparent_bg = st.sidebar.checkbox("Transparent Background", True)
show_legend = st.sidebar.checkbox("Show Legend", True)
show_colorbar = st.sidebar.checkbox("Show Colorbar", True)
scientific_notation = st.sidebar.checkbox("Scientific Notation for Ticks", False)

threshold_style = st.sidebar.selectbox(
    "Threshold Line Style",
    ["--", "-", "-.", ":"],
    index=0
)

# 1. 50+ Colormaps
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

default_cmap_idx = available_colormaps.index('inferno') if 'inferno' in available_colormaps else 0
selected_cmap = st.sidebar.selectbox("Colormap", available_colormaps, index=default_cmap_idx)

# ==========================================
# 19. Custom Parameter Count (Dynamic List)
# ==========================================
st.sidebar.header("Parameter Configuration")
st.sidebar.markdown("Define control variables, ranges, thresholds, and reference values.")

if 'params' not in st.session_state:
    st.session_state.params = [
        {'name': '$A_s$ (J/mol)', 'min': 0.0, 'max': 1.0, 'thresh': 0.05, 'dir': 'lower', 'ref': 0.3},
        {'name': '$\kappa$ ($10^{-10}$ J/m)', 'min': 1.0, 'max': 10.0, 'thresh': 4.0, 'dir': 'lower', 'ref': 6.0},
        {'name': '$U$ (V)', 'min': -0.5, 'max': -0.2, 'thresh': -0.33, 'dir': 'lower', 'ref': -0.30},
        {'name': '$\psi$ ($10^{-3}$ s$^{-1}$)', 'min': 0.0, 'max': 5.0, 'thresh': 1.5, 'dir': 'upper', 'ref': 1.0}
    ]

def add_param():
    st.session_state.params.append({'name': 'New Param', 'min': 0.0, 'max': 1.0, 'thresh': 0.5, 'dir': 'lower', 'ref': 0.5})

def remove_param(idx):
    if len(st.session_state.params) > 1:
        st.session_state.params.pop(idx)

for i, p in enumerate(st.session_state.params):
    st.sidebar.markdown(f"**Parameter {i+1}**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        p['name'] = st.text_input("Name", p['name'], key=f"name_{i}")
        p['min'] = st.number_input("Min", value=p['min'], key=f"min_{i}", format="%.3f")
        p['thresh'] = st.number_input("Threshold", value=p['thresh'], key=f"thresh_{i}", format="%.3f")
    with col2:
        p['max'] = st.number_input("Max", value=p['max'], key=f"max_{i}", format="%.3f")
        p['dir'] = st.selectbox("Safe Region", options=['> Threshold (Lower bound)', '< Threshold (Upper bound)'], 
                                index=0 if p['dir'] == 'lower' else 1, key=f"dir_{i}")
        p['ref'] = st.number_input("Reference", value=p['ref'], key=f"ref_{i}", format="%.3f")
    
    st.sidebar.button("Remove", on_click=remove_param, args=(i,), key=f"rem_{i}")

st.sidebar.button("Add Parameter", on_click=add_param)

# ==========================================
# Plotting Function
# ==========================================
def create_plot(params_data, title_font, label_font, tick_font, threshold_font, reference_font,
                fig_width, fig_height, line_width, trajectory_width, marker_size, dpi,
                show_grid, dark_theme, transparent_bg, show_legend, show_colorbar, threshold_style, selected_cmap, scientific_notation):
    
    # 10. Dark/light themes
    if dark_theme:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
        
    plt.rcParams.update({'font.size': tick_font})

    N = len(params_data)
    x_pos = np.arange(N)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 9. Colormap-driven safe/unsafe coloring
    cmap = plt.get_cmap(selected_cmap)
    safe_color = cmap(0.85)
    unsafe_color = cmap(0.15)
    reference_color = cmap(0.55)
    
    # Normalize reference and thresholds to [0, 1]
    for p in params_data:
        p['r_norm'] = max(0.0, min(1.0, (p['ref'] - p['min']) / (p['max'] - p['min'])))
        p['t_norm'] = max(0.0, min(1.0, (p['thresh'] - p['min']) / (p['max'] - p['min'])))

    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if show_grid:
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    for i, p in enumerate(params_data):
        axis_color = 'black' if not dark_theme else 'white'
        
        # 15. Parameter axis thickness control
        ax.plot([x_pos[i], x_pos[i]], [0, 1], color=axis_color, linewidth=line_width, zorder=1)
        ax.text(x_pos[i], 1.05, p['name'], ha='center', va='bottom', fontsize=label_font, fontweight='bold')
        
        # Draw ticks and physical value labels
        ticks = np.linspace(0, 1, 5)
        tick_vals = np.linspace(p['min'], p['max'], 5)
        for t, val in zip(ticks, tick_vals):
            ax.plot([x_pos[i]-0.02, x_pos[i]+0.02], [t, t], color=axis_color, linewidth=1)
            # 18. Scientific notation support
            val_str = f'{val:.2e}' if scientific_notation else f'{val:.2f}'
            ax.text(x_pos[i]-0.08, t, val_str, ha='right', va='center', fontsize=tick_font)
            
        # Draw safe and unsafe shaded regions
        if p['dir'] == 'lower': # Safe is > threshold
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], p['t_norm'], 1, color=safe_color, alpha=0.3, zorder=0)
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], 0, p['t_norm'], color=unsafe_color, alpha=0.2, zorder=0)
            ax.plot([x_pos[i]-0.08, x_pos[i]+0.08], [p['t_norm'], p['t_norm']], color=unsafe_color, linestyle=threshold_style, linewidth=1.5, zorder=2)
            ax.text(x_pos[i]+0.12, p['t_norm'], f"{p['thresh']}", va='center', ha='left', color=unsafe_color, fontsize=threshold_font, fontweight='bold')
        else: # Safe is < threshold
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], 0, p['t_norm'], color=safe_color, alpha=0.3, zorder=0)
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], p['t_norm'], 1, color=unsafe_color, alpha=0.2, zorder=0)
            ax.plot([x_pos[i]-0.08, x_pos[i]+0.08], [p['t_norm'], p['t_norm']], color=unsafe_color, linestyle=threshold_style, linewidth=1.5, zorder=2)
            ax.text(x_pos[i]+0.12, p['t_norm'], f"{p['thresh']}", va='center', ha='left', color=unsafe_color, fontsize=threshold_font, fontweight='bold')

    # 16. Reference trajectory style customization
    ax.plot(x_pos, [p['r_norm'] for p in params_data], color=reference_color, linewidth=trajectory_width, marker='o', 
            markersize=marker_size, markeredgecolor=axis_color, markeredgewidth=1.0, zorder=5, label='Reference Safe Point')
    
    for i, p in enumerate(params_data):
        ax.text(x_pos[i], p['r_norm'] + 0.03, f"{p['ref']}", ha='center', va='bottom', 
                fontsize=reference_font, color=reference_color, fontweight='bold')

    # 3. Integer control-variable numbering on bottom axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i+1) for i in range(N)], fontsize=tick_font, fontweight='bold')
    ax.set_xlabel("Control Variable Number", fontsize=label_font, fontweight='bold')

    # Better Figure Title
    ax.set_title("Stability Map for Dendrite Suppression", fontsize=title_font, fontweight='bold', pad=25)
    
    if show_legend:
        ax.legend(loc='upper right', fontsize=tick_font)
        
    # 17. Colorbar display
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Normalized Stability Scale", fontsize=label_font)

    plt.tight_layout()
    return fig

# ==========================================
# Generate Plot
# ==========================================
fig = create_plot(
    st.session_state.params, title_font, label_font, tick_font, threshold_font, reference_font,
    fig_width, fig_height, line_width, trajectory_width, marker_size, dpi,
    show_grid, dark_theme, transparent_bg, show_legend, show_colorbar, threshold_style, selected_cmap, scientific_notation
)

st.pyplot(fig)

# ==========================================
# 11, 12. Multi-format Export with Transparent Background
# ==========================================
st.sidebar.header("Export Options")
st.markdown("Download the generated figure in your preferred format for publication.")

pdf_buf = io.BytesIO()
png_buf = io.BytesIO()
svg_buf = io.BytesIO()

fig.savefig(pdf_buf, format='pdf', dpi=dpi, transparent=transparent_bg)
fig.savefig(png_buf, format='png', dpi=dpi, transparent=transparent_bg)
fig.savefig(svg_buf, format='svg', transparent=transparent_bg)

col1, col2, col3 = st.columns(3)
with col1:
    st.download_button(
        label="📥 Download PDF",
        data=pdf_buf.getvalue(),
        file_name="stability_map.pdf",
        mime="application/pdf"
    )
with col2:
    st.download_button(
        label="📥 Download PNG",
        data=png_buf.getvalue(),
        file_name="stability_map.png",
        mime="image/png"
    )
with col3:
    st.download_button(
        label="📥 Download SVG",
        data=svg_buf.getvalue(),
        file_name="stability_map.svg",
        mime="image/svg+xml"
    )
