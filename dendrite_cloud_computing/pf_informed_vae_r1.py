import io
import os
import re
from pathlib import Path

import streamlit as st
from PIL import Image
import imageio.v2 as imageio


# ============================================================
# Configuration
# ============================================================
DEFAULT_DATA_ROOT = "/data/phasefield"
DATA_ROOT = Path(os.getenv("DATA_ROOT", DEFAULT_DATA_ROOT))

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# Utilities
# ============================================================
_num_re = re.compile(r"(\d+)")

def natural_sort_key(path: Path):
    """Sort filenames using numeric order if numbers exist."""
    parts = _num_re.split(path.name)
    key = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p.lower())
    return key


@st.cache_data(ttl=5)
def scan_simulations(data_root: str):
    """Scan raw/ and model/ directories."""
    root = Path(data_root)
    result = {"raw": [], "model": []}

    for category in result.keys():
        d = root / category
        if d.exists():
            result[category] = sorted(
                [p for p in d.iterdir() if p.is_dir()],
                key=lambda x: x.name.lower()
            )
    return result


@st.cache_data(ttl=60, show_spinner=False)
def list_frames(sim_dir: str):
    """List all image frames inside a simulation directory."""
    d = Path(sim_dir)
    frames = [
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]
    frames.sort(key=natural_sort_key)
    return [str(p) for p in frames]


@st.cache_data(ttl=300, show_spinner=False)
def load_image(path: str):
    return Image.open(path).convert("RGB")


def build_gif(frame_paths, fps: int):
    duration = 1.0 / max(1, fps)
    frames = [load_image(p) for p in frame_paths]
    buffer = io.BytesIO()
    imageio.mimsave(buffer, frames, format="GIF", duration=duration)
    return buffer.getvalue()


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(
    page_title="Phase-Field Simulation Viewer",
    layout="wide"
)

st.title("Phase-Field Simulation Animation Viewer")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Simulation Selection")

    st.caption("Data root directory")
    st.code(str(DATA_ROOT))

    simulations = scan_simulations(str(DATA_ROOT))

    sim_type_label = st.radio(
        "Simulation type",
        options=[
            "Original simulation (raw)",
            "Model prediction (model)"
        ],
        index=0
    )
    sim_type = "raw" if "raw" in sim_type_label else "model"

    sim_dirs = simulations.get(sim_type, [])
    if not sim_dirs:
        st.error(f"No simulations found under: {DATA_ROOT / sim_type}")
        st.stop()

    sim_name = st.selectbox(
        "Select simulation",
        options=[p.name for p in sim_dirs]
    )

    sim_dir = str((DATA_ROOT / sim_type / sim_name).resolve())
    frame_paths = list_frames(sim_dir)

    if not frame_paths:
        st.warning("No image frames found in this simulation directory.")
        st.stop()

    total_frames = len(frame_paths)
    st.write(f"Total frames: **{total_frames}**")

    if total_frames > 1:
        start, end = st.slider(
            "Frame range",
            0, total_frames - 1,
            (0, min(total_frames - 1, 200))
        )
    else:
        start, end = 0, 0

    stride = st.slider(
        "Frame stride (take every k-th frame)",
        min_value=1,
        max_value=50,
        value=1
    )

    fps = st.slider(
        "Frames per second (FPS)",
        min_value=1,
        max_value=60,
        value=15
    )

    max_frames = st.number_input(
        "Maximum output frames (safety limit)",
        min_value=10,
        max_value=5000,
        value=400,
        step=10
    )


# ---------------- Frame selection ----------------
selected_frames = frame_paths[start:end + 1:stride]
if len(selected_frames) > max_frames:
    selected_frames = selected_frames[:max_frames]
    st.warning(f"Frame count truncated to {max_frames} frames.")


# ---------------- Main layout ----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("First Frame Preview")
    st.image(load_image(selected_frames[0]), use_container_width=True)
    st.caption(selected_frames[0])

    st.subheader("Frame-by-frame Viewer")
    idx = st.slider("Current frame index", 0, len(selected_frames) - 1, 0)
    st.image(load_image(selected_frames[idx]), use_container_width=True)
    st.caption(Path(selected_frames[idx]).name)

with col2:
    st.subheader("Animated GIF")

    if "gif_bytes" not in st.session_state:
        st.session_state.gif_bytes = None

    if st.button("Generate / Refresh GIF"):
        with st.spinner("Generating GIF..."):
            st.session_state.gif_bytes = build_gif(
                selected_frames,
                int(fps)
            )

    if st.session_state.gif_bytes:
        size_mb = len(st.session_state.gif_bytes) / 1024 / 1024
        st.success(
            f"GIF generated | "
            f"{len(selected_frames)} frames | "
            f"{fps} FPS | "
            f"{size_mb:.2f} MB"
        )

        st.image(st.session_state.gif_bytes)

        st.download_button(
            "Download GIF",
            data=st.session_state.gif_bytes,
            file_name=f"{sim_type}_{sim_name}_{start}_{end}_s{stride}_fps{fps}.gif",
            mime="image/gif"
        )


# ---------------- Debug info ----------------
st.divider()
st.subheader("Current Configuration")
st.json({
    "DATA_ROOT": str(DATA_ROOT),
    "simulation_type": sim_type,
    "simulation_name": sim_name,
    "simulation_directory": sim_dir,
    "total_frames": total_frames,
    "selected_range": [start, end],
    "stride": stride,
    "selected_frames": len(selected_frames),
    "fps": fps,
})
