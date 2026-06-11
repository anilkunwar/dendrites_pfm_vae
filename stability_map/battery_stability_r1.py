import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io

# Page configuration
st.set_page_config(layout="wide")
st.title("🔋 Interactive Stability Map for Dendrite Suppression")
st.markdown("Adjust the parameter thresholds and reference design points below to explore the safe operating window. You can download the high-resolution PDF directly for your LaTeX manuscript.")

# Sidebar inputs for interactivity
st.sidebar.header("Control Parameters")

# Default values based on your manuscript
default_names = ['$A_s$\n(J/mol)', '$\kappa$\n($10^{-10}$ J/m)', '$U$\n(V)', '$\psi$\n($10^{-3}$ s$^{-1}$)']
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

# Create interactive widgets for each parameter
for i in range(4):
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

# Plotting function
def create_plot(params, mins, maxs, thresholds, directions, ref_point):
    ref_norm = [(ref_point[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(params))]
    thresh_norm = [(thresholds[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(params))]

    fig, ax = plt.subplots(figsize=(10, 6))
    N = len(params)
    x_pos = np.arange(N)

    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(N):
        ax.plot([x_pos[i], x_pos[i]], [0, 1], color='black', linewidth=1.5, zorder=1)
        ax.text(x_pos[i], 1.05, params[i], ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ticks = np.linspace(0, 1, 5)
        tick_vals = np.linspace(mins[i], maxs[i], 5)
        for t, val in zip(ticks, tick_vals):
            ax.plot([x_pos[i]-0.02, x_pos[i]+0.02], [t, t], color='black', linewidth=1)
            ax.text(x_pos[i]-0.08, t, f'{val:.2f}', ha='right', va='center', fontsize=10)
            
        # Clip normalized values to [0, 1] to prevent plotting errors if thresholds are out of bounds
        t_norm = max(0, min(1, thresh_norm[i]))
        
        if directions[i] == 'lower':
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], t_norm, 1, color='green', alpha=0.3, zorder=0)
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], 0, t_norm, color='red', alpha=0.2, zorder=0)
            ax.plot([x_pos[i]-0.08, x_pos[i]+0.08], [t_norm, t_norm], color='red', linestyle='--', linewidth=1.5, zorder=2)
            ax.text(x_pos[i]+0.12, t_norm, f'{thresholds[i]}', va='center', ha='left', color='red', fontsize=10, fontweight='bold')
        else:
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], 0, t_norm, color='green', alpha=0.3, zorder=0)
            ax.fill_between([x_pos[i]-0.08, x_pos[i]+0.08], t_norm, 1, color='red', alpha=0.2, zorder=0)
            ax.plot([x_pos[i]-0.08, x_pos[i]+0.08], [t_norm, t_norm], color='red', linestyle='--', linewidth=1.5, zorder=2)
            ax.text(x_pos[i]+0.12, t_norm, f'{thresholds[i]}', va='center', ha='left', color='red', fontsize=10, fontweight='bold')

    # Clip reference norm to [0, 1]
    r_norm = [max(0, min(1, rn)) for rn in ref_norm]
    ax.plot(x_pos, r_norm, color='blue', linewidth=2.5, zorder=3, marker='o', markersize=6, markerfacecolor='blue')
    for i in range(N):
        ax.text(x_pos[i], r_norm[i] + 0.03, f'{ref_point[i]}', ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')

    plt.title('Stability Map for Dendrite Suppression', fontsize=14, pad=20)
    plt.tight_layout()
    return fig

# Generate and display the plot in the Streamlit app
fig = create_plot(params, mins, maxs, thresholds, directions, ref_point)
st.pyplot(fig)

# Add a download button for the high-res PDF
buf = io.BytesIO()
fig.savefig(buf, format="pdf", bbox_inches='tight')
byte_im = buf.getvalue()

st.download_button(
   label="📥 Download High-Res PDF for LaTeX",
   data=byte_im,
   file_name="stability_map.pdf",
   mime="application/pdf"
)
