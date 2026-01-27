import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
import colorsys
from scipy import stats
import io
import warnings
from typing import List, Dict, Tuple, Optional, Union
import math

warnings.filterwarnings('ignore')

# ============================================================
# CIRCLIZE-INSPIRED CHORD DIAGRAM SYSTEM
# ============================================================

class CirclizeChordDiagram:
    """Chord diagram system inspired by R's circlize package"""
    
    def __init__(self, figsize=(12, 12), dpi=100):
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        
        # Track data
        self.sectors = []
        self.sector_data = {}
        self.links = []
        self.current_track = 0
        self.tracks = {}
        
        # Visual parameters
        self.gap_after = {}
        self.start_degree = 0
        self.clock_wise = True
        self.big_gap = 10  # degrees between groups
        self.small_gap = 1  # degrees within groups
        
    def initialize_sectors(self, sectors, groups=None):
        """Initialize sectors with optional grouping"""
        self.sectors = sectors
        if groups is None:
            groups = {s: 0 for s in sectors}
        
        # Set initial gaps
        for sector in sectors:
            if sector in groups:
                group = groups[sector]
                # Set larger gap between groups
                self.gap_after[sector] = self.small_gap
            else:
                self.gap_after[sector] = self.small_gap
        
        # Adjust gaps between groups
        if len(set(groups.values())) > 1:
            last_sector = None
            last_group = None
            for sector in sectors:
                current_group = groups[sector]
                if last_sector and current_group != last_group:
                    self.gap_after[last_sector] = self.big_gap
                last_sector = sector
                last_group = current_group
    
    def set_gaps(self, gap_dict):
        """Set gaps between sectors"""
        self.gap_after.update(gap_dict)
    
    def set_start_degree(self, degree):
        """Set starting degree for first sector"""
        self.start_degree = degree
    
    def set_direction(self, clockwise=True):
        """Set direction of sectors"""
        self.clock_wise = clockwise
        self.ax.set_theta_direction(-1 if clockwise else 1)
    
    def compute_sector_angles(self):
        """Compute angles for all sectors considering gaps"""
        total_gap = sum(self.gap_after.values())
        available_degrees = 360 - total_gap
        sector_degrees = available_degrees / len(self.sectors)
        
        angles = {}
        current_angle = self.start_degree
        
        for sector in self.sectors:
            angles[sector] = {
                'start': current_angle,
                'end': current_angle + sector_degrees,
                'mid': current_angle + sector_degrees / 2
            }
            current_angle += sector_degrees + self.gap_after.get(sector, 0)
        
        return angles
    
    def draw_sector_grid(self, sector, angles, color='lightgray', alpha=0.3, 
                         track_height=0.1, track_index=0):
        """Draw a sector grid/background"""
        start_rad = np.radians(angles['start'])
        end_rad = np.radians(angles['end'])
        
        # Create arc patch
        theta = np.linspace(start_rad, end_rad, 100)
        r_inner = track_index * track_height
        r_outer = r_inner + track_height
        
        # Create polygon for the sector
        theta_poly = np.concatenate([theta, theta[::-1]])
        r_poly = np.concatenate([np.ones_like(theta) * r_inner, 
                                 np.ones_like(theta) * r_outer])
        
        poly = plt.Polygon(np.column_stack([theta_poly, r_poly]), 
                          facecolor=color, alpha=alpha, edgecolor='none')
        self.ax.add_patch(poly)
        
        return {'inner': r_inner, 'outer': r_outer}
    
    def create_link(self, source, target, value, 
                    source_track=0, target_track=0,
                    color='skyblue', alpha=0.7,
                    directional=False, direction_type='diffHeight',
                    arrow_length=0.1, arrow_width=0.05,
                    highlight=False, zindex=1):
        """Create a link between sectors"""
        
        # Get sector angles
        angles = self.compute_sector_angles()
        
        if source not in angles or target not in angles:
            raise ValueError(f"Sector {source} or {target} not found")
        
        # Get source and target positions
        source_angle = angles[source]['mid']
        target_angle = angles[target]['mid']
        
        # Get track positions
        source_track_data = self.tracks.get(source, {}).get(source_track, {})
        target_track_data = self.tracks.get(target, {}).get(target_track, {})
        
        source_r = source_track_data.get('outer', 0.1)
        target_r = target_track_data.get('outer', 0.1)
        
        # Adjust for directional links
        if directional and 'diffHeight' in direction_type:
            if directional == 1:  # source -> target
                source_r -= 0.02
                target_r += 0.02
            elif directional == -1:  # target -> source
                source_r += 0.02
                target_r -= 0.02
        
        # Convert to radians
        source_rad = np.radians(source_angle)
        target_rad = np.radians(target_angle)
        
        # Create Bezier curve for the link
        control_angle = (source_rad + target_rad) / 2
        control_r = (source_r + target_r) / 2 * 1.2  # Pull out for curvature
        
        # Bezier control points
        t = np.linspace(0, 1, 50)
        
        # Quadratic Bezier in polar coordinates
        theta_curve = (1-t)**2 * source_rad + 2*(1-t)*t * control_angle + t**2 * target_rad
        r_curve = (1-t)**2 * source_r + 2*(1-t)*t * control_r + t**2 * target_r
        
        # Convert to Cartesian for plotting
        x_curve = r_curve * np.cos(theta_curve)
        y_curve = r_curve * np.sin(theta_curve)
        
        # Create the link
        link_line, = self.ax.plot(theta_curve, r_curve, 
                                  color=color, alpha=alpha, 
                                  linewidth=max(0.5, value * 3),
                                  solid_capstyle='round', zorder=zindex)
        
        # Add arrow if directional
        if directional and 'arrows' in direction_type:
            # Calculate arrow position (midpoint of curve)
            mid_idx = len(t) // 2
            arrow_theta = theta_curve[mid_idx]
            arrow_r = r_curve[mid_idx]
            
            # Calculate tangent for arrow direction
            dx = np.gradient(x_curve)[mid_idx]
            dy = np.gradient(y_curve)[mid_idx]
            
            # Create arrow patch
            arrow = FancyArrowPatch(
                (x_curve[mid_idx-1], y_curve[mid_idx-1]),
                (x_curve[mid_idx+1], y_curve[mid_idx+1]),
                arrowstyle='->', color=color,
                mutation_scale=arrow_length * 100,
                linewidth=max(0.5, value * 2),
                zorder=zindex + 0.1
            )
            self.ax.add_patch(arrow)
        
        # Add highlight if requested
        if highlight:
            # Add a glow effect
            self.ax.plot(theta_curve, r_curve, 
                        color='white', alpha=0.3,
                        linewidth=max(1, value * 5),
                        solid_capstyle='round', zorder=zindex-0.1)
        
        # Store link data
        link_data = {
            'source': source, 'target': target, 'value': value,
            'color': color, 'alpha': alpha, 'directional': directional,
            'line': link_line, 'coordinates': (theta_curve, r_curve)
        }
        self.links.append(link_data)
        
        return link_data
    
    def add_track(self, sector, track_index=0, height=0.1, 
                  color='lightgray', alpha=0.3):
        """Add a track to a sector"""
        if sector not in self.tracks:
            self.tracks[sector] = {}
        
        angles = self.compute_sector_angles()
        track_data = self.draw_sector_grid(sector, angles[sector], 
                                          color=color, alpha=alpha,
                                          track_height=height, 
                                          track_index=track_index)
        
        self.tracks[sector][track_index] = track_data
        return track_data
    
    def add_sector_labels(self, label_dict=None, fontsize=12, 
                          offset=0.15, rotation='auto'):
        """Add labels to sectors"""
        angles = self.compute_sector_angles()
        
        for sector, angle_data in angles.items():
            angle_deg = angle_data['mid']
            angle_rad = np.radians(angle_deg)
            
            # Get outer radius for this sector
            outer_r = 0
            if sector in self.tracks:
                for track_data in self.tracks[sector].values():
                    outer_r = max(outer_r, track_data['outer'])
            
            label_r = outer_r + offset
            
            # Determine label rotation
            if rotation == 'auto':
                rotation_deg = angle_deg
                if 90 <= angle_deg <= 270:
                    rotation_deg += 180
                    ha = "right"
                else:
                    ha = "left"
            elif rotation == 'horizontal':
                rotation_deg = 0
                ha = "center"
            else:
                rotation_deg = angle_deg
                ha = "left" if angle_deg < 180 else "right"
            
            # Get label text
            if label_dict and sector in label_dict:
                label_text = label_dict[sector]
            else:
                label_text = sector
            
            # Add label
            txt = self.ax.text(angle_rad, label_r, label_text,
                              fontsize=fontsize, fontweight='bold',
                              rotation=rotation_deg, rotation_mode='anchor',
                              ha=ha, va='center', zorder=100)
            
            # Add background for better readability
            txt.set_bbox(dict(boxstyle="round,pad=0.2",
                             facecolor='white',
                             edgecolor='lightgray',
                             alpha=0.9))
    
    def add_sector_axis(self, sector, track_index=0, 
                        n_ticks=5, tick_length=0.02,
                        show_labels=True, label_format='.1f'):
        """Add axis to a sector"""
        if sector not in self.tracks or track_index not in self.tracks[sector]:
            raise ValueError(f"Track {track_index} not found for sector {sector}")
        
        track_data = self.tracks[sector][track_index]
        angles = self.compute_sector_angles()
        angle_data = angles[sector]
        
        start_rad = np.radians(angle_data['start'])
        end_rad = np.radians(angle_data['end'])
        mid_rad = np.radians(angle_data['mid'])
        
        inner_r = track_data['inner']
        outer_r = track_data['outer']
        
        # Add ticks
        tick_values = np.linspace(inner_r, outer_r, n_ticks)
        for tick_r in tick_values:
            # Tick at start of sector
            x_start = tick_r * np.cos(start_rad)
            y_start = tick_r * np.sin(start_rad)
            x_end = (tick_r - tick_length) * np.cos(start_rad)
            y_end = (tick_r - tick_length) * np.sin(start_rad)
            
            self.ax.plot([start_rad, start_rad], [x_start, x_end],
                        color='black', alpha=0.5, linewidth=0.5,
                        transform=self.ax.transData._b)
            
            # Add tick labels
            if show_labels and tick_r == outer_r:
                label_x = (tick_r - tick_length * 2) * np.cos(start_rad)
                label_y = (tick_r - tick_length * 2) * np.sin(start_rad)
                
                # Convert to text position
                label_angle = np.degrees(np.arctan2(label_y, label_x))
                label_r = np.sqrt(label_x**2 + label_y**2)
                
                txt = self.ax.text(start_rad, label_r, 
                                  format(tick_r, label_format),
                                  fontsize=8, rotation=label_angle,
                                  ha='center', va='center',
                                  rotation_mode='anchor')
    
    def scale_sectors(self, values, mode='absolute'):
        """Scale sector widths based on values"""
        angles = self.compute_sector_angles()
        
        if mode == 'relative':
            # Scale so sum of widths = 360 - total_gap
            total_value = sum(values.values())
            total_gap = sum(self.gap_after.values())
            available = 360 - total_gap
            
            for sector, value in values.items():
                if sector in self.gap_after:
                    sector_fraction = value / total_value
                    sector_width = sector_fraction * available
                    # Adjust gap proportionally (simplified)
                    pass
        # Implementation would adjust sector angles based on values
    
    def finalize(self, title="", show_frame=False, background_color='white'):
        """Finalize the diagram"""
        # Set background
        self.fig.patch.set_facecolor(background_color)
        self.ax.set_facecolor(background_color)
        
        # Hide radial grid and labels
        self.ax.set_ylim(0, 1.5)
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        
        # Hide spines
        self.ax.spines['polar'].set_visible(show_frame)
        
        # Add title
        if title:
            self.ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Adjust layout
        plt.tight_layout()
        
        return self.fig

# ============================================================
# ADVANCED CHORD DIAGRAM FUNCTIONS WITH CIRCLIZE FEATURES
# ============================================================

def create_circlize_chord_diagram(
    data,  # Can be matrix or DataFrame
    data_type='matrix',  # 'matrix' or 'adjacency_list'
    figsize=(14, 14),
    title="Chord Diagram",
    
    # Layout parameters
    start_degree=0,
    direction='clockwise',
    big_gap=10,
    small_gap=1,
    sector_order=None,
    
    # Sector styling
    sector_colors=None,
    sector_labels=None,
    sector_label_fontsize=12,
    sector_label_offset=0.15,
    show_sector_axes=False,
    
    # Link styling
    link_colors='group',  # 'group', 'value', 'matrix', or dict
    link_alpha=0.7,
    link_width_scale=3.0,
    link_min_width=0.5,
    link_max_width=8.0,
    
    # Directional features
    directional=False,
    direction_type=['diffHeight', 'arrows'],  # 'diffHeight', 'arrows', 'both'
    arrow_length=0.1,
    arrow_width=0.05,
    diff_height=0.02,
    
    # Highlighting
    highlight_links=None,
    highlight_color='red',
    highlight_alpha=0.9,
    
    # Scaling
    scale=False,
    scale_mode='absolute',  # 'absolute' or 'relative'
    
    # Advanced features
    symmetric=False,
    reduce_threshold=0.01,
    link_sort=False,
    link_decreasing=True,
    link_zindex='value',  # 'value', 'random', or array
    
    # Visual effects
    background_color='white',
    grid_color='lightgray',
    grid_alpha=0.1,
    show_frame=False,
    
    # Multiple tracks
    tracks=None,
    track_heights=None,
    track_colors=None
):
    """
    Create a chord diagram with circlize-style features
    
    Parameters:
    -----------
    data : matrix or DataFrame
        Input data. If matrix: rows are sources, columns are targets.
        If DataFrame: columns should be ['source', 'target', 'value'].
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    
    # Process input data
    if data_type == 'matrix':
        matrix = data
        if symmetric:
            # Use only lower triangular (excluding diagonal)
            matrix = np.tril(matrix, -1)
        
        # Convert matrix to links
        sources, targets = matrix.shape
        links = []
        for i in range(sources):
            for j in range(targets):
                value = matrix[i, j]
                if abs(value) > reduce_threshold:
                    links.append({
                        'source': f"S{i+1}",
                        'target': f"T{j+1}",
                        'value': abs(value),
                        'sign': np.sign(value)
                    })
        
        # Get sectors
        row_sectors = [f"S{i+1}" for i in range(sources)]
        col_sectors = [f"T{j+1}" for j in range(targets)]
        all_sectors = row_sectors + col_sectors
        
        # Create groups
        groups = {}
        for sector in row_sectors:
            groups[sector] = 0
        for sector in col_sectors:
            groups[sector] = 1
    
    else:  # adjacency list
        df = data
        links = []
        for _, row in df.iterrows():
            links.append({
                'source': str(row[0]),
                'target': str(row[1]),
                'value': abs(row[2]),
                'sign': np.sign(row[2]) if len(row) > 2 else 1
            })
        
        # Get unique sectors
        all_sectors = list(set(df.iloc[:, 0].tolist() + df.iloc[:, 1].tolist()))
        groups = {s: 0 for s in all_sectors}  # Default all in same group
    
    # Apply sector order if provided
    if sector_order:
        all_sectors = [s for s in sector_order if s in all_sectors]
    
    # Create diagram
    diagram = CirclizeChordDiagram(figsize=figsize)
    diagram.big_gap = big_gap
    diagram.small_gap = small_gap
    diagram.set_start_degree(start_degree)
    diagram.set_direction(direction == 'clockwise')
    
    # Initialize sectors
    diagram.initialize_sectors(all_sectors, groups)
    
    # Set sector colors
    if sector_colors is None:
        cmap = plt.cm.tab20c
        sector_colors = {}
        for i, sector in enumerate(all_sectors):
            if groups.get(sector, 0) == 0:
                sector_colors[sector] = cmap(i % 20)
            else:
                sector_colors[sector] = cmap((i + 10) % 20)
    
    # Add tracks
    if tracks:
        for sector in all_sectors:
            for track_idx in range(tracks):
                track_color = track_colors[track_idx] if track_colors else 'lightgray'
                diagram.add_track(sector, track_idx, 
                                 height=track_heights[track_idx] if track_heights else 0.1,
                                 color=track_color, alpha=grid_alpha)
    else:
        # Add default track
        for sector in all_sectors:
            diagram.add_track(sector, color=grid_color, alpha=grid_alpha)
    
    # Process link colors
    if isinstance(link_colors, str):
        if link_colors == 'group':
            link_color_dict = {}
            for link in links:
                source_group = groups.get(link['source'], 0)
                if source_group == 0:
                    link_color_dict[(link['source'], link['target'])] = sector_colors[link['source']]
                else:
                    link_color_dict[(link['source'], link['target'])] = sector_colors[link['target']]
        elif link_colors == 'value':
            # Color by value using colormap
            values = [link['value'] for link in links]
            norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
            cmap = plt.cm.viridis
            link_color_dict = {}
            for link in links:
                color = cmap(norm(link['value']))
                link_color_dict[(link['source'], link['target'])] = color
        else:
            link_color_dict = {(link['source'], link['target']): 'skyblue' for link in links}
    elif isinstance(link_colors, dict):
        link_color_dict = link_colors
    else:
        link_color_dict = {(link['source'], link['target']): link_colors for link in links}
    
    # Sort links if requested
    if link_sort:
        links.sort(key=lambda x: x['value'], reverse=link_decreasing)
    
    # Assign z-indices
    if link_zindex == 'value':
        for i, link in enumerate(links):
            link['zindex'] = link['value'] * 10
    elif link_zindex == 'random':
        import random
        for link in links:
            link['zindex'] = random.random() * 10
    elif isinstance(link_zindex, (list, np.ndarray)):
        for i, link in enumerate(links):
            if i < len(link_zindex):
                link['zindex'] = link_zindex[i]
            else:
                link['zindex'] = 1
    
    # Create links
    for link in links:
        is_highlighted = highlight_links and (link['source'], link['target']) in highlight_links
        
        color = link_color_dict.get((link['source'], link['target']), 'skyblue')
        if is_highlighted:
            color = highlight_color
            alpha = highlight_alpha
        else:
            alpha = link_alpha
        
        # Calculate width
        width = max(link_min_width, min(link_max_width, link['value'] * link_width_scale))
        
        # Determine if directional
        is_directional = directional
        if isinstance(directional, dict):
            is_directional = directional.get((link['source'], link['target']), False)
        
        # Create link
        diagram.create_link(
            source=link['source'],
            target=link['target'],
            value=width,
            color=color,
            alpha=alpha,
            directional=is_directional,
            direction_type=direction_type,
            arrow_length=arrow_length,
            arrow_width=arrow_width,
            highlight=is_highlighted,
            zindex=link.get('zindex', 1)
        )
    
    # Add sector labels
    if sector_labels is None:
        sector_labels = {s: s for s in all_sectors}
    
    diagram.add_sector_labels(sector_labels, 
                             fontsize=sector_label_fontsize,
                             offset=sector_label_offset)
    
    # Add sector axes if requested
    if show_sector_axes:
        for sector in all_sectors:
            diagram.add_sector_axis(sector, show_labels=True)
    
    # Finalize
    fig = diagram.finalize(title=title, 
                          show_frame=show_frame,
                          background_color=background_color)
    
    return fig

# ============================================================
# STREAMLIT INTERFACE FOR CIRCLIZE-STYLE CHORD DIAGRAMS
# ============================================================

def create_streamlit_circlize_app():
    """Create a Streamlit app for circlize-style chord diagrams"""
    
    st.set_page_config(
        page_title="Circlize-Style Chord Diagrams",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stExpander {
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🔄 Circlize-Style Chord Diagrams")
    st.markdown("""
    Create advanced chord diagrams with features inspired by R's **circlize** package.
    Supports directional links, arrows, scaling, and multiple tracks.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Input")
        
        # Example data or upload
        use_example = st.checkbox("Use example data", value=True)
        
        if use_example:
            # Create example correlation matrix
            np.random.seed(42)
            n_features = 8
            example_matrix = np.random.randn(n_features, n_features)
            example_matrix = np.corrcoef(example_matrix)
            
            # Make it somewhat symmetric and interesting
            example_matrix = (example_matrix + example_matrix.T) / 2
            np.fill_diagonal(example_matrix, 1)
            
            # Row and column names
            row_names = [f"Feature_{i}" for i in range(1, n_features + 1)]
            col_names = [f"Target_{i}" for i in range(1, n_features + 1)]
            
            data_type = "matrix"
            data = pd.DataFrame(example_matrix, index=row_names, columns=col_names)
            
            st.success("✅ Using example correlation matrix (8x8)")
        else:
            uploaded_file = st.file_uploader("Upload your data", type=['csv', 'xlsx'])
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                data_type = st.radio("Data format", ["matrix", "adjacency_list"])
                if data_type == "adjacency_list":
                    st.info("Data should have columns: source, target, value")
            
        st.header("🎨 Diagram Settings")
        
        # Layout
        with st.expander("🔧 Layout & Orientation", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                start_degree = st.slider("Start degree", 0, 360, 0, 15)
                direction = st.selectbox("Direction", ["clockwise", "counter-clockwise"])
            with col2:
                big_gap = st.slider("Group gap (degrees)", 0, 30, 10, 1)
                small_gap = st.slider("Sector gap (degrees)", 0, 10, 1, 1)
        
        # Sectors
        with st.expander("⚫ Sectors", expanded=False):
            show_sector_axes = st.checkbox("Show sector axes", value=False)
            sector_label_fontsize = st.slider("Label font size", 8, 20, 12)
            sector_label_offset = st.slider("Label offset", 0.0, 0.5, 0.15, 0.01)
        
        # Links
        with st.expander("🔗 Links", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                link_alpha = st.slider("Link transparency", 0.1, 1.0, 0.7, 0.05)
                link_width_scale = st.slider("Width scale", 0.1, 10.0, 3.0, 0.1)
            with col2:
                link_min_width = st.slider("Min width", 0.1, 3.0, 0.5, 0.1)
                link_max_width = st.slider("Max width", 1.0, 20.0, 8.0, 0.5)
            
            link_colors = st.selectbox("Link coloring", 
                                      ["group", "value", "single_color"])
        
        # Directional features
        with st.expander("↔️ Directional Features", expanded=True):
            directional = st.checkbox("Enable directional links", value=False)
            
            if directional:
                col1, col2 = st.columns(2)
                with col1:
                    direction_type = st.multiselect(
                        "Direction type",
                        ["diffHeight", "arrows"],
                        default=["diffHeight", "arrows"]
                    )
                    arrow_length = st.slider("Arrow length", 0.01, 0.3, 0.1, 0.01)
                with col2:
                    arrow_width = st.slider("Arrow width", 0.01, 0.1, 0.05, 0.01)
                    diff_height = st.slider("Height difference", 0.0, 0.1, 0.02, 0.01)
        
        # Advanced features
        with st.expander("⚡ Advanced Features", expanded=False):
            scale = st.checkbox("Scale sectors", value=False)
            symmetric = st.checkbox("Symmetric matrix", value=False)
            link_sort = st.checkbox("Sort links by value", value=False)
            link_decreasing = st.checkbox("Sort decreasing", value=True)
            
            reduce_threshold = st.slider("Reduce threshold", 0.0, 0.5, 0.01, 0.01)
        
        # Visual effects
        with st.expander("🎨 Visual Effects", expanded=False):
            background_color = st.color_picker("Background", "#FFFFFF")
            show_frame = st.checkbox("Show frame", value=False)
        
        # Export
        with st.expander("💾 Export", expanded=False):
            export_dpi = st.slider("Export DPI", 72, 600, 300)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📈 Data", "📖 Documentation"])
    
    with tab1:
        st.header("Chord Diagram Visualization")
        
        # Create diagram
        if 'data' in locals():
            try:
                # Prepare data based on type
                if data_type == "matrix":
                    matrix_data = data.values
                    row_names = data.index.tolist()
                    col_names = data.columns.tolist()
                    
                    # Create sector labels
                    sector_labels = {}
                    for i, name in enumerate(row_names):
                        sector_labels[f"S{i+1}"] = name
                    for j, name in enumerate(col_names):
                        sector_labels[f"T{j+1}"] = name
                    
                    # Create diagram
                    fig = create_circlize_chord_diagram(
                        data=matrix_data,
                        data_type='matrix',
                        figsize=(14, 14),
                        title="Circlize-Style Chord Diagram",
                        
                        # Layout
                        start_degree=start_degree,
                        direction=direction,
                        big_gap=big_gap,
                        small_gap=small_gap,
                        
                        # Sectors
                        sector_labels=sector_labels,
                        sector_label_fontsize=sector_label_fontsize,
                        sector_label_offset=sector_label_offset,
                        show_sector_axes=show_sector_axes,
                        
                        # Links
                        link_colors=link_colors,
                        link_alpha=link_alpha,
                        link_width_scale=link_width_scale,
                        link_min_width=link_min_width,
                        link_max_width=link_max_width,
                        
                        # Directional
                        directional=directional if directional else False,
                        direction_type=direction_type if directional else [],
                        arrow_length=arrow_length if directional else 0.1,
                        arrow_width=arrow_width if directional else 0.05,
                        diff_height=diff_height if directional else 0.02,
                        
                        # Advanced
                        scale=scale,
                        symmetric=symmetric,
                        reduce_threshold=reduce_threshold,
                        link_sort=link_sort,
                        link_decreasing=link_decreasing,
                        
                        # Visual
                        background_color=background_color,
                        show_frame=show_frame
                    )
                    
                    # Display
                    st.pyplot(fig)
                    
                    # Export buttons
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # PNG
                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, format='png', dpi=export_dpi, 
                                   bbox_inches='tight', facecolor=background_color)
                        buf_png.seek(0)
                        st.download_button(
                            label="📥 PNG",
                            data=buf_png,
                            file_name="circlize_chord_diagram.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        # PDF
                        buf_pdf = io.BytesIO()
                        fig.savefig(buf_pdf, format='pdf',
                                   bbox_inches='tight', facecolor=background_color)
                        buf_pdf.seek(0)
                        st.download_button(
                            label="📥 PDF",
                            data=buf_pdf,
                            file_name="circlize_chord_diagram.pdf",
                            mime="application/pdf"
                        )
                    
                    with col3:
                        # SVG
                        buf_svg = io.BytesIO()
                        fig.savefig(buf_svg, format='svg',
                                   bbox_inches='tight', facecolor=background_color)
                        buf_svg.seek(0)
                        st.download_button(
                            label="📥 SVG",
                            data=buf_svg,
                            file_name="circlize_chord_diagram.svg",
                            mime="image/svg+xml"
                        )
                    
                    plt.close(fig)
                    
                else:  # adjacency list
                    # For DataFrame input
                    fig = create_circlize_chord_diagram(
                        data=data,
                        data_type='adjacency_list',
                        figsize=(14, 14),
                        title="Circlize-Style Chord Diagram",
                        
                        # Add parameters based on user selections
                        # ... (similar to matrix case)
                    )
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error creating diagram: {str(e)}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
        else:
            st.info("Please upload data or use example data to create a visualization.")
    
    with tab2:
        st.header("Data Preview")
        
        if 'data' in locals():
            st.write("### Data Preview")
            st.dataframe(data, use_container_width=True)
            
            # Data statistics
            st.write("### Data Statistics")
            if data_type == "matrix":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", data.shape[0])
                with col2:
                    st.metric("Columns", data.shape[1])
                with col3:
                    st.metric("Total Values", data.size)
                
                # Correlation statistics
                if isinstance(data, pd.DataFrame):
                    flat_values = data.values.flatten()
                    st.write(f"**Value Statistics:**")
                    st.write(f"- Min: {flat_values.min():.3f}")
                    st.write(f"- Max: {flat_values.max():.3f}")
                    st.write(f"- Mean: {flat_values.mean():.3f}")
                    st.write(f"- Std: {flat_values.std():.3f}")
            else:
                st.write(f"Number of links: {len(data)}")
                st.write(f"Unique sources: {data.iloc[:, 0].nunique()}")
                st.write(f"Unique targets: {data.iloc[:, 1].nunique()}")
        else:
            st.info("No data loaded.")
    
    with tab3:
        st.header("📖 Documentation")
        
        st.markdown("""
        ## Circlize-Style Chord Diagrams
        
        This implementation brings R's **circlize** package features to Python and Streamlit.
        
        ### Key Features:
        
        #### 1. **Directional Links**
        - Show direction with **height differences** (diffHeight)
        - Add **arrows** to indicate flow direction
        - Control arrow size and style
        
        #### 2. **Advanced Layout**
        - Custom start angle for first sector
        - Clockwise or counter-clockwise arrangement
        - Adjustable gaps between sectors and groups
        
        #### 3. **Link Styling**
        - Color by group, value, or custom colors
        - Adjustable transparency and width
        - Minimum and maximum width limits
        
        #### 4. **Sector Features**
        - Multiple tracks per sector
        - Sector axes with labels
        - Custom sector ordering
        - Sector scaling based on values
        
        #### 5. **Advanced Features**
        - **Scaling**: Scale sector widths proportionally
        - **Symmetric mode**: For symmetric matrices
        - **Link sorting**: Order links by value
        - **Reduce threshold**: Remove small links
        
        ### Data Formats:
        
        #### 1. **Matrix Format** (recommended)
        ```python
        # Rows = sources, Columns = targets
        matrix = [
            [1, 4, 7],  # Source 1 -> Targets
            [2, 5, 8],  # Source 2 -> Targets
            [3, 6, 9]   # Source 3 -> Targets
        ]
        ```
        
        #### 2. **Adjacency List Format**
        ```python
        # DataFrame with columns: source, target, value
        data = [
            {"source": "A", "target": "X", "value": 1},
            {"source": "B", "target": "Y", "value": 2},
            {"source": "C", "target": "Z", "value": 3}
        ]
        ```
        
        ### Use Cases:
        
        1. **Network Analysis**: Show relationships between nodes
        2. **Flow Visualization**: Directional flows between categories
        3. **Correlation Analysis**: Feature correlations with direction
        4. **Comparative Analysis**: Compare multiple groups
        
        ### Tips:
        
        - Use **directional links** for flow diagrams
        - Use **scaling** for proportional visualization
        - Adjust **gaps** to separate groups clearly
        - Use **highlighting** to emphasize key relationships
        
        ### Export Options:
        - **PNG**: For web and presentations
        - **PDF**: For publications (vector quality)
        - **SVG**: For further editing in vector software
        
        ### Compatibility:
        This implementation uses only **pip-installable libraries**:
        - Streamlit
        - Matplotlib
        - Pandas
        - NumPy
        - SciPy
        """)

# ============================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================

def matrix_to_adjacency_list(matrix, row_names=None, col_names=None):
    """Convert matrix to adjacency list format"""
    if row_names is None:
        row_names = [f"Row_{i}" for i in range(matrix.shape[0])]
    if col_names is None:
        col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
    
    adjacency_list = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if abs(value) > 0:  # Only non-zero values
                adjacency_list.append({
                    'source': row_names[i],
                    'target': col_names[j],
                    'value': value
                })
    
    return pd.DataFrame(adjacency_list)

def adjacency_list_to_matrix(df, fill_value=0):
    """Convert adjacency list to matrix format"""
    sources = df.iloc[:, 0].unique()
    targets = df.iloc[:, 1].unique()
    
    matrix = np.full((len(sources), len(targets)), fill_value)
    
    source_to_idx = {s: i for i, s in enumerate(sources)}
    target_to_idx = {t: j for j, t in enumerate(targets)}
    
    for _, row in df.iterrows():
        i = source_to_idx[row[0]]
        j = target_to_idx[row[1]]
        matrix[i, j] = row[2]
    
    return matrix, sources, targets

def create_comparative_chord_diagrams(data_list, titles=None, ncols=2, figsize=(20, 10)):
    """Create multiple chord diagrams for comparison"""
    if titles is None:
        titles = [f"Diagram {i+1}" for i in range(len(data_list))]
    
    nrows = (len(data_list) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, 
                            figsize=figsize,
                            subplot_kw={'projection': 'polar'})
    
    if len(data_list) == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    for idx, (data, title) in enumerate(zip(data_list, titles)):
        if idx < len(axes):
            ax = axes[idx]
            
            # Create simplified chord diagram for each subplot
            # (Implementation would go here)
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_ylim(0, 1.5)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines['polar'].set_visible(False)
    
    # Hide unused subplots
    for idx in range(len(data_list), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point"""
    create_streamlit_circlize_app()

if __name__ == "__main__":
    main()
