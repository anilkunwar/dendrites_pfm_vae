import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(layout="wide")
st.title("📊 Row-wise Chord Diagram Visualizer (15 Features)")

# -----------------------------
# Helper functions
# -----------------------------
def get_colormap(name, n):
    cmap = cm.get_cmap(name, n)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]

def make_chord_sankey(values, labels, cmap_name, label_size, opacity, scale):
    n = len(values)

    # normalize values for link weights
    weights = values * scale

    # create full connection matrix (i -> j)
    source, target, value = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                source.append(i)
                target.append(j)
                value.append(weights[i] * weights[j])

    colors = get_colormap(cmap_name, n)

    fig = go.Figure(
        go.Sankey(
            arrangement="freeform",
            node=dict(
                pad=15,
                thickness=18,
                label=labels,
                color=colors,
                line=dict(color="black", width=0.5),
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=[
                    colors[s] for s in source
                ],
                opacity=opacity,
            ),
        )
    )

    fig.update_layout(
        font=dict(size=label_size),
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

colormap_list = sorted([
    "viridis", "plasma", "inferno", "magma", "cividis",
    "jet", "rainbow", "turbo", "cool", "hot",
    "spring", "summer", "autumn", "winter",
    "cubehelix", "gnuplot", "gnuplot2",
    "terrain", "ocean", "brg", "hsv",
    "tab10", "tab20", "tab20b", "tab20c",
    "Set1", "Set2", "Set3", "Paired",
    "Accent", "Dark2", "Pastel1", "Pastel2"
])

cmap_name = st.sidebar.selectbox("Colormap", colormap_list, index=2)
label_size = st.sidebar.slider("Label size", 8, 24, 14)
opacity = st.sidebar.slider("Link opacity", 0.1, 1.0, 0.6)
scale = st.sidebar.slider("Weight scaling", 0.1, 5.0, 1.0)

normalize = st.sidebar.checkbox("Normalize each row", value=True)

row_range = st.sidebar.slider("Rows to plot", 0, 200, (0, 5))
cols_per_row = st.sidebar.selectbox("Figures per row", [1, 2, 3], index=1)

# -----------------------------
# Main logic
# -----------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # features starting from 't'
    start_col = df.columns.get_loc("t")
    features = df.columns[start_col:start_col + 15].tolist()

    st.success(f"Using features:\n{features}")

    rows = range(row_range[0], min(row_range[1] + 1, len(df)))
    cols = st.columns(cols_per_row)
    col_idx = 0

    for i in rows:
        values = df.loc[i, features].values.astype(float)

        if normalize:
            values = values / (values.max() + 1e-12)

        fig = make_chord_sankey(
            values=values,
            labels=features,
            cmap_name=cmap_name,
            label_size=label_size,
            opacity=opacity,
            scale=scale,
        )

        with cols[col_idx]:
            st.markdown(f"### Row {i}")
            st.plotly_chart(fig, use_container_width=True)

        col_idx = (col_idx + 1) % cols_per_row

else:
    st.info("⬅️ Upload a CSV file to begin.")
