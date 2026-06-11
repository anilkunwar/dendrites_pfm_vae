import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# =============================================================================
# USER CONFIGURATION – adjust these values to match your manuscript
# =============================================================================

# Parameter definitions: name, min, max, threshold, direction ('lower' or 'upper')
params = [
    {"name": "$A_s$ (J/mol)",       "min": 0.0,  "max": 1.0,  "thresh": 0.05,  "dir": "lower"},
    {"name": "$\kappa$ ($10^{-10}$ J/m)", "min": 1.0,  "max": 10.0, "thresh": 4.0,   "dir": "lower"},
    {"name": "$U$ (V)",             "min": -0.5, "max": -0.2, "thresh": -0.33, "dir": "lower"},
    {"name": "$\psi$ ($10^{-3}$ s$^{-1}$)", "min": 0.0,  "max": 5.0,  "thresh": 1.5,   "dir": "upper"},
]

# Reference point (example safe design)
ref_point = [0.3, 6.0, -0.30, 1.0]

# Figure appearance
title_font      = 20
label_font      = 16
tick_font       = 12
threshold_font  = 12
reference_font  = 12

fig_width       = 12   # inches
fig_height      = 7    # inches
line_width      = 2.0
trajectory_width= 3.0
marker_size     = 8
dpi             = 600
show_grid       = True
dark_theme      = False
show_legend     = True
show_colorbar   = True

# Colormap (choose from matplotlib's large set)
colormap_name   = "viridis"   # alternatives: 'plasma', 'inferno', 'turbo', 'cividis', ...
threshold_style = "--"        # line style for threshold lines (--, -, -., :)

# Axis labeling: use integer numbers on bottom
use_axis_numbers = True

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

# Set journal-quality rcParams
rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight'
})

if dark_theme:
    plt.style.use('dark_background')
else:
    plt.style.use('default')

cmap = plt.get_cmap(colormap_name)
safe_color   = cmap(0.85)
unsafe_color = cmap(0.15)
ref_color    = cmap(0.55)

N = len(params)
x_pos = np.arange(N)

# Normalise thresholds and reference point to [0,1] based on each axis limits
mins   = [p["min"] for p in params]
maxs   = [p["max"] for p in params]
thresh_norm = []
ref_norm    = []
for i, p in enumerate(params):
    t = p["thresh"]
    r = ref_point[i]
    # clip to axis range to avoid plotting errors
    thresh_norm.append(max(0, min(1, (t - mins[i]) / (maxs[i] - mins[i]))))
    ref_norm.append(max(0, min(1, (r - mins[i]) / (maxs[i] - mins[i]))))

# Create figure
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Axis limits and aesthetics
ax.set_ylim(-0.05, 1.15)
ax.set_xlim(-0.5, N - 0.5)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

if show_grid:
    ax.grid(axis='y', linestyle=':', alpha=0.6)

# Draw vertical axes and tick labels
for i in range(N):
    # Axis line
    ax.plot([x_pos[i], x_pos[i]], [0, 1], color='black', linewidth=line_width, zorder=1)
    # Parameter name
    ax.text(x_pos[i], 1.05, params[i]["name"], ha='center', va='bottom', fontsize=label_font, fontweight='bold')
    # Tick marks and labels
    ticks = np.linspace(0, 1, 5)
    tick_vals = np.linspace(mins[i], maxs[i], 5)
    for t, val in zip(ticks, tick_vals):
        ax.plot([x_pos[i]-0.02, x_pos[i]+0.02], [t, t], color='black', linewidth=1)
        ax.text(x_pos[i]-0.1, t, f'{val:.2f}', ha='right', va='center', fontsize=tick_font)

# Shade safe/unsafe regions
for i in range(N):
    t_norm = thresh_norm[i]
    if params[i]["dir"] == "lower":
        # Safe = above threshold
        ax.fill_between([x_pos[i]-0.1, x_pos[i]+0.1], t_norm, 1, color=safe_color, alpha=0.3, zorder=0)
        ax.fill_between([x_pos[i]-0.1, x_pos[i]+0.1], 0, t_norm, color=unsafe_color, alpha=0.2, zorder=0)
        ax.plot([x_pos[i]-0.1, x_pos[i]+0.1], [t_norm, t_norm], color='red', linestyle=threshold_style, linewidth=1.5, zorder=2)
        ax.text(x_pos[i]+0.12, t_norm, f'{params[i]["thresh"]}', va='center', ha='left', color='red', fontsize=threshold_font, fontweight='bold')
    else:
        # Safe = below threshold (upper bound)
        ax.fill_between([x_pos[i]-0.1, x_pos[i]+0.1], 0, t_norm, color=safe_color, alpha=0.3, zorder=0)
        ax.fill_between([x_pos[i]-0.1, x_pos[i]+0.1], t_norm, 1, color=unsafe_color, alpha=0.2, zorder=0)
        ax.plot([x_pos[i]-0.1, x_pos[i]+0.1], [t_norm, t_norm], color='red', linestyle=threshold_style, linewidth=1.5, zorder=2)
        ax.text(x_pos[i]+0.12, t_norm, f'{params[i]["thresh"]}', va='center', ha='left', color='red', fontsize=threshold_font, fontweight='bold')

# Reference trajectory (polyline)
ax.plot(x_pos, ref_norm, color=ref_color, linewidth=trajectory_width,
        marker='o', markersize=marker_size, markeredgecolor='black', markeredgewidth=1.0,
        zorder=5, label='Reference safe design' if show_legend else "")
for i in range(N):
    ax.text(x_pos[i], ref_norm[i] + 0.03, f'{ref_point[i]}', ha='center', va='bottom',
            fontsize=reference_font, color=ref_color, fontweight='bold')

# Bottom axis with integer numbers (optional)
if use_axis_numbers:
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i+1) for i in range(N)], fontsize=tick_font, fontweight='bold')
    ax.set_xlabel("Control variable number", fontsize=label_font, fontweight='bold')
else:
    ax.set_xticks([])

# Title
ax.set_title("Stability Map for Dendrite Suppression", fontsize=title_font, fontweight='bold', pad=25)

# Legend
if show_legend:
    ax.legend(loc='upper right', fontsize=tick_font)

# Colorbar (optional)
if show_colorbar:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.7)
    cbar.set_label("Normalized stability scale", fontsize=label_font)

plt.tight_layout()

# Save figures (choose format)
fig.savefig("stability_map.pdf", dpi=dpi, format='pdf')
fig.savefig("stability_map.png", dpi=dpi, format='png')
fig.savefig("stability_map.svg", format='svg')
print("Figure saved as PDF, PNG, and SVG.")

# If running in a notebook, show the plot
# plt.show()
