import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
import colorsys
from scipy import stats
import io
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
import math
import traceback
import random
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

CHORD_DIAGRAM_DEFAULTS = {
    'figsize': (14, 14),
    'dpi': 100,
    'big_gap': 10.0,      # degrees between groups
    'small_gap': 1.0,     # degrees within groups
    'start_degree': 0,
    'clockwise': True,
    'track_height': 0.12,
    'grid_alpha': 0.15,
    'link_min_width': 0.8,
    'link_max_width': 10.0,
    'link_width_scale': 3.5,
    'link_alpha': 0.75,
    'arrow_length': 0.12,
    'arrow_width': 0.06,
    'diff_height': 0.03,
    'label_offset': 0.18,
    'label_fontsize': 13,
    'background_color': '#FFFFFF',
    'grid_color': '#E0E0E0',
    'highlight_alpha': 0.95,
    'reduce_threshold': 0.005
}

# ============================================================================
# CORE CHORD DIAGRAM ENGINE (Circlize-Inspired)
# ============================================================================

class CirclizeChordDiagram:
    """
    Advanced chord diagram renderer inspired by R's circlize package.
    
    Features:
        - Polar coordinate layout with customizable gaps and direction
        - Multi-track sector architecture
        - Directional links with height differentiation and arrows
        - Group-based sector organization
        - Comprehensive styling controls
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 12), dpi: int = 100):
        """
        Initialize chord diagram canvas.
        
        Parameters
        ----------
        figsize : tuple
            Figure dimensions (width, height) in inches
        dpi : int
            Resolution in dots per inch
        """
        self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_theta_zero_location("N")  # 0° at top
        self.ax.set_theta_direction(-1)       # Clockwise by default
        
        # Data structures
        self.sectors: List[str] = []
        self.sector_data: Dict[str, Dict] = {}
        self.links: List[Dict] = []
        self.tracks: Dict[str, Dict[int, Dict]] = {}
        self.groups: Dict[str, int] = {}
        
        # Layout parameters
        self.gap_after: Dict[str, float] = {}
        self.start_degree: float = 0.0
        self.clockwise: bool = True
        self.big_gap: float = CHORD_DIAGRAM_DEFAULTS['big_gap']
        self.small_gap: float = CHORD_DIAGRAM_DEFAULTS['small_gap']
        self.sector_angles: Dict[str, Dict[str, float]] = {}
        
        # Visual parameters
        self.track_height: float = CHORD_DIAGRAM_DEFAULTS['track_height']
        self.grid_alpha: float = CHORD_DIAGRAM_DEFAULTS['grid_alpha']
        self.background_color: str = CHORD_DIAGRAM_DEFAULTS['background_color']
        
    def initialize_sectors(self, sectors: List[str], groups: Optional[Dict[str, int]] = None) -> None:
        """
        Initialize sectors with optional grouping for visual separation.
        
        Parameters
        ----------
        sectors : list of str
            List of sector identifiers in display order
        groups : dict, optional
            Mapping from sector name to group ID (creates larger gaps between groups)
        """
        self.sectors = sectors
        self.groups = groups if groups else {s: 0 for s in sectors}
        
        # Initialize default small gaps for all sectors
        self.gap_after = {sector: self.small_gap for sector in sectors}
        
        # Apply larger gaps between different groups
        if len(set(self.groups.values())) > 1:
            for i in range(len(sectors) - 1):
                current_sector = sectors[i]
                next_sector = sectors[i + 1]
                if self.groups[current_sector] != self.groups[next_sector]:
                    self.gap_after[current_sector] = self.big_gap
    
    def set_gaps(self, gap_dict: Dict[str, float]) -> None:
        """Override default gaps between specific sectors."""
        self.gap_after.update(gap_dict)
    
    def set_start_degree(self, degree: float) -> None:
        """Set starting angle (in degrees) for the first sector."""
        self.start_degree = degree % 360
    
    def set_direction(self, clockwise: bool = True) -> None:
        """Set drawing direction (clockwise or counter-clockwise)."""
        self.clockwise = clockwise
        self.ax.set_theta_direction(-1 if clockwise else 1)
    
    def compute_sector_angles(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate precise angular positions for all sectors considering gaps.
        
        Returns
        -------
        dict
            Mapping from sector name to {start, end, mid} angles in degrees
        """
        total_gap = sum(self.gap_after.get(sector, self.small_gap) for sector in self.sectors)
        available_degrees = 360.0 - total_gap
        
        # Equal width sectors (enhancement: could support variable widths)
        sector_width = available_degrees / len(self.sectors)
        
        angles = {}
        current_angle = self.start_degree
        
        for sector in self.sectors:
            angles[sector] = {
                'start': current_angle % 360,
                'end': (current_angle + sector_width) % 360,
                'mid': (current_angle + sector_width / 2) % 360
            }
            current_angle += sector_width + self.gap_after.get(sector, self.small_gap)
        
        self.sector_angles = angles
        return angles
    
    def draw_sector_grid(self, sector: str, angles: Dict[str, float], 
                        color: str = 'lightgray', alpha: float = 0.3,
                        track_height: float = 0.1, track_index: int = 0) -> Dict[str, float]:
        """
        Render a single sector track as a curved rectangular patch.
        
        Parameters
        ----------
        sector : str
            Sector identifier
        angles : dict
            Angular boundaries with 'start', 'end', 'mid' keys (degrees)
        color : str
            Fill color for the track
        alpha : float
            Transparency level [0-1]
        track_height : float
            Radial height of the track
        track_index : int
            Track position (0 = innermost)
        
        Returns
        -------
        dict
            Radial boundaries {'inner': r_inner, 'outer': r_outer}
        """
        start_rad = np.radians(angles['start'])
        end_rad = np.radians(angles['end'])
        
        # Handle sectors crossing 0° boundary
        if end_rad < start_rad and abs(end_rad - start_rad) < np.pi:
            end_rad += 2 * np.pi
        
        # Create smooth arc
        theta = np.linspace(start_rad, end_rad, 100)
        r_inner = track_index * track_height
        r_outer = r_inner + track_height
        
        # Build polygon vertices in polar coordinates
        theta_poly = np.concatenate([theta, theta[::-1]])
        r_poly = np.concatenate([np.full_like(theta, r_inner), 
                                np.full_like(theta, r_outer)])
        
        # Convert to Cartesian for patch creation
        x = r_poly * np.cos(theta_poly)
        y = r_poly * np.sin(theta_poly)
        vertices = np.column_stack([x, y])
        
        # Create and add patch
        poly = plt.Polygon(vertices, facecolor=color, alpha=alpha, 
                          edgecolor='none', zorder=0.5)
        self.ax.add_patch(poly)
        
        return {'inner': r_inner, 'outer': r_outer, 'mid_angle': np.radians(angles['mid'])}
    
    def create_link(self, source: str, target: str, value: float,
                   source_track: int = 0, target_track: int = 0,
                   color: Union[str, Tuple] = 'skyblue', alpha: float = 0.7,
                   directional: Union[bool, int] = False, 
                   direction_type: Union[str, List[str]] = 'diffHeight',
                   arrow_length: float = 0.1, arrow_width: float = 0.05,
                   highlight: bool = False, zindex: float = 1.0) -> Dict[str, Any]:
        """
        Create a curved link (chord) between two sectors with advanced styling.
        
        Parameters
        ----------
        source, target : str
            Source and target sector identifiers
        value : float
            Link weight (controls visual width)
        source_track, target_track : int
            Track indices for connection points
        color : str or RGB tuple
            Link color
        alpha : float
            Transparency [0-1]
        directional : bool or int
            Direction indicator: False=undirected, 1=source→target, -1=target→source
        direction_type : str or list
            Visual direction cues: 'diffHeight', 'arrows', or both
        arrow_length, arrow_width : float
            Arrow dimensions (normalized units)
        highlight : bool
            Apply visual emphasis (glow effect)
        zindex : float
            Rendering order (higher = on top)
        
        Returns
        -------
        dict
            Link metadata including matplotlib artist reference
        """
        # Validate sectors exist
        if source not in self.sector_angles or target not in self.sector_angles:
            raise ValueError(f"Sector '{source}' or '{target}' not initialized. "
                           f"Available sectors: {list(self.sector_angles.keys())}")
        
        # Get angular positions
        source_angle = self.sector_angles[source]['mid']
        target_angle = self.sector_angles[target]['mid']
        
        # Get radial positions from track data
        source_r = self.tracks.get(source, {}).get(source_track, {}).get('outer', 0.12)
        target_r = self.tracks.get(target, {}).get(target_track, {}).get('outer', 0.12)
        
        # Apply directional height offset if requested
        if directional and isinstance(directional, int):
            if 'diffHeight' in direction_type or direction_type == 'diffHeight':
                offset = CHORD_DIAGRAM_DEFAULTS['diff_height']
                if directional == 1:   # source → target
                    source_r -= offset
                    target_r += offset
                elif directional == -1:  # target → source
                    source_r += offset
                    target_r -= offset
        
        # Convert to radians
        source_rad = np.radians(source_angle)
        target_rad = np.radians(target_angle)
        
        # Handle angular wrap-around for smoother curves
        if abs(target_rad - source_rad) > np.pi:
            if target_rad > source_rad:
                target_rad -= 2 * np.pi
            else:
                source_rad -= 2 * np.pi
        
        # Create quadratic Bezier curve in polar space
        control_angle = (source_rad + target_rad) / 2
        control_r = max(source_r, target_r) * 1.4  # Curve "bulge" factor
        
        # Parameterize curve
        t = np.linspace(0, 1, 60)
        theta_curve = (1 - t)**2 * source_rad + 2 * (1 - t) * t * control_angle + t**2 * target_rad
        r_curve = (1 - t)**2 * source_r + 2 * (1 - t) * t * control_r + t**2 * target_r
        
        # Normalize angles to [0, 2π)
        theta_curve = theta_curve % (2 * np.pi)
        
        # Render base link
        line, = self.ax.plot(theta_curve, r_curve,
                           color=color, alpha=alpha,
                           linewidth=max(CHORD_DIAGRAM_DEFAULTS['link_min_width'], 
                                       min(CHORD_DIAGRAM_DEFAULTS['link_max_width'], 
                                           value * CHORD_DIAGRAM_DEFAULTS['link_width_scale'])),
                           solid_capstyle='round', 
                           solid_joinstyle='round',
                           zorder=zindex)
        
        # Add directional arrows if requested
        if directional and ('arrows' in direction_type or direction_type == 'arrows'):
            # Position arrow at 60% along the curve for visual clarity
            arrow_idx = int(len(t) * 0.6)
            arrow_theta = theta_curve[arrow_idx]
            arrow_r = r_curve[arrow_idx]
            
            # Compute tangent direction for arrow orientation
            if arrow_idx > 0 and arrow_idx < len(t) - 1:
                dx = r_curve[arrow_idx + 1] * np.cos(theta_curve[arrow_idx + 1]) - \
                     r_curve[arrow_idx - 1] * np.cos(theta_curve[arrow_idx - 1])
                dy = r_curve[arrow_idx + 1] * np.sin(theta_curve[arrow_idx + 1]) - \
                     r_curve[arrow_idx - 1] * np.sin(theta_curve[arrow_idx - 1])
                arrow_angle = np.arctan2(dy, dx)
                
                # Create arrow using FancyArrowPatch in Cartesian space
                x = arrow_r * np.cos(arrow_theta)
                y = arrow_r * np.sin(arrow_theta)
                arrow_length_abs = arrow_length * 0.15
                
                arrow = FancyArrowPatch(
                    (x - dx/10, y - dy/10),
                    (x + dx/10, y + dy/10),
                    arrowstyle=f'->,head_width={arrow_width*8},head_length={arrow_length*6}',
                    color=color,
                    alpha=alpha * 1.2,  # Slightly more opaque arrows
                    linewidth=1.5,
                    mutation_scale=20,
                    zorder=zindex + 0.5
                )
                self.ax.add_patch(arrow)
        
        # Apply highlight effect (glow)
        if highlight:
            glow_width = line.get_linewidth() * 2.2
            self.ax.plot(theta_curve, r_curve,
                       color='white', alpha=0.45,
                       linewidth=glow_width,
                       solid_capstyle='round',
                       zorder=zindex - 0.5)
            self.ax.plot(theta_curve, r_curve,
                       color=color, alpha=min(1.0, alpha * 1.5),
                       linewidth=line.get_linewidth() * 1.3,
                       solid_capstyle='round',
                       zorder=zindex + 0.2)
        
        # Store link metadata
        link_data = {
            'source': source,
            'target': target,
            'value': value,
            'color': color,
            'alpha': alpha,
            'directional': directional,
            'line': line,
            'coordinates': (theta_curve.copy(), r_curve.copy()),
            'zindex': zindex
        }
        self.links.append(link_data)
        return link_data
    
    def add_track(self, sector: str, track_index: int = 0, height: Optional[float] = None,
                 color: str = 'lightgray', alpha: Optional[float] = None) -> Dict[str, float]:
        """
        Add a concentric track to a sector for layered visualizations.
        
        Parameters
        ----------
        sector : str
            Target sector identifier
        track_index : int
            Track position (0 = innermost)
        height : float, optional
            Radial height of track (uses default if None)
        color : str
            Track fill color
        alpha : float, optional
            Transparency (uses default if None)
        
        Returns
        -------
        dict
            Track geometry data {'inner', 'outer', 'mid_angle'}
        """
        if sector not in self.tracks:
            self.tracks[sector] = {}
        
        if height is None:
            height = self.track_height
        if alpha is None:
            alpha = self.grid_alpha
        
        if sector not in self.sector_angles:
            self.compute_sector_angles()
        
        track_data = self.draw_sector_grid(
            sector, 
            self.sector_angles[sector],
            color=color,
            alpha=alpha,
            track_height=height,
            track_index=track_index
        )
        self.tracks[sector][track_index] = track_data
        return track_data
    
    def add_sector_labels(self, label_dict: Optional[Dict[str, str]] = None, 
                         fontsize: Optional[int] = None,
                         offset: Optional[float] = None, 
                         rotation: str = 'auto') -> None:
        """
        Add readable labels to sectors with automatic positioning.
        
        Parameters
        ----------
        label_dict : dict, optional
            Custom label mapping {sector: label_text}
        fontsize : int, optional
            Font size (uses default if None)
        offset : float, optional
            Radial distance from outer track edge
        rotation : str
            'auto' (follows circle), 'horizontal', or fixed angle
        """
        if not self.sector_angles:
            self.compute_sector_angles()
        
        if fontsize is None:
            fontsize = CHORD_DIAGRAM_DEFAULTS['label_fontsize']
        if offset is None:
            offset = CHORD_DIAGRAM_DEFAULTS['label_offset']
        
        for sector, angle_data in self.sector_angles.items():
            angle_deg = angle_data['mid']
            angle_rad = np.radians(angle_deg)
            
            # Determine outer radius from tracks
            outer_r = 0.0
            if sector in self.tracks:
                for track_data in self.tracks[sector].values():
                    outer_r = max(outer_r, track_data['outer'])
            
            label_r = outer_r + offset
            
            # Smart text alignment based on position
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
                rotation_deg = float(rotation)
                ha = "center"
            
            # Get label text
            label_text = label_dict.get(sector, sector) if label_dict else sector
            
            # Add label with background for readability
            txt = self.ax.text(
                angle_rad, label_r, label_text,
                fontsize=fontsize, 
                fontweight='bold',
                fontfamily='sans-serif',
                rotation=rotation_deg, 
                rotation_mode='anchor',
                ha=ha, 
                va='center',
                zorder=1000,
                color='#2D3748'  # Dark gray for readability
            )
            
            # Add subtle background box
            txt.set_bbox(dict(
                boxstyle="round,pad=0.25",
                facecolor='white',
                edgecolor='none',
                alpha=0.85,
                boxstyle=FancyBboxPatch(
                    (0, 0), 1, 1,
                    boxstyle="round,pad=0.25",
                    mutation_scale=0.5
                ).get_boxstyle()
            ))
    
    def finalize(self, title: str = "", show_frame: bool = False, 
                background_color: Optional[str] = None) -> plt.Figure:
        """
        Finalize diagram appearance and layout.
        
        Parameters
        ----------
        title : str
            Diagram title
        show_frame : bool
            Display polar coordinate frame
        background_color : str, optional
            Canvas background color
        
        Returns
        -------
        matplotlib.figure.Figure
            Rendered figure object
        """
        if background_color is None:
            background_color = self.background_color
        
        # Apply background colors
        self.fig.patch.set_facecolor(background_color)
        self.ax.set_facecolor(background_color)
        
        # Configure axes appearance
        max_radius = 1.5
        for sector_tracks in self.tracks.values():
            if sector_tracks:
                max_radius = max(max_radius, max(t['outer'] for t in sector_tracks.values()))
        
        self.ax.set_ylim(0, max_radius * 1.25)  # Extra space for labels
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        
        # Frame visibility
        self.ax.spines['polar'].set_visible(show_frame)
        if show_frame:
            self.ax.spines['polar'].set_color('#94A3B8')
            self.ax.spines['polar'].set_linewidth(1.2)
        
        # Add title with styling
        if title:
            self.ax.set_title(
                title, 
                fontsize=18, 
                fontweight='bold', 
                pad=25,
                color='#1E293B',
                fontfamily='sans-serif'
            )
        
        # Remove margins
        self.fig.tight_layout(pad=1.5)
        return self.fig


# ============================================================================
# HIGH-LEVEL API FUNCTIONS
# ============================================================================

def create_circlize_chord_diagram(
    data: Any,
    data_type: str = 'matrix',
    figsize: Tuple[int, int] = (14, 14),
    title: str = "Chord Diagram",
    # Layout parameters
    start_degree: float = 0,
    direction: str = 'clockwise',
    big_gap: float = 10.0,
    small_gap: float = 1.0,
    sector_order: Optional[List[str]] = None,
    # Sector styling
    sector_colors: Optional[Dict[str, str]] = None,
    sector_labels: Optional[Dict[str, str]] = None,
    sector_label_fontsize: int = 13,
    sector_label_offset: float = 0.18,
    show_sector_axes: bool = False,
    # Link styling
    link_colors: Union[str, Dict[Tuple[str, str], str]] = 'group',
    link_alpha: float = 0.75,
    link_width_scale: float = 3.5,
    link_min_width: float = 0.8,
    link_max_width: float = 10.0,
    # Directional features
    directional: Union[bool, Dict[Tuple[str, str], int]] = False,
    direction_type: Union[str, List[str]] = ['diffHeight', 'arrows'],
    arrow_length: float = 0.12,
    arrow_width: float = 0.06,
    diff_height: float = 0.03,
    # Highlighting
    highlight_links: Optional[List[Tuple[str, str]]] = None,
    highlight_color: str = '#EF4444',  # Red-500
    highlight_alpha: float = 0.95,
    # Scaling
    scale: bool = False,
    scale_mode: str = 'absolute',
    # Advanced features
    symmetric: bool = False,
    reduce_threshold: float = 0.005,
    link_sort: bool = True,
    link_decreasing: bool = True,
    link_zindex: Union[str, List[float]] = 'value',
    # Visual effects
    background_color: str = '#FFFFFF',
    grid_color: str = '#E2E8F0',  # Slate-200
    grid_alpha: float = 0.15,
    show_frame: bool = False,
    # Multiple tracks
    tracks: int = 1,
    track_heights: Optional[List[float]] = None,
    track_colors: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create publication-quality chord diagrams with circlize-inspired features.
    
    Parameters
    ----------
    data : array-like or DataFrame
        Input data matrix (sources × targets) or adjacency list DataFrame
    data_type : str
        'matrix' for 2D arrays, 'adjacency_list' for DataFrame with [source, target, value]
    figsize : tuple
        Figure dimensions in inches
    title : str
        Diagram title
    
    Returns
    -------
    matplotlib.figure.Figure
        Fully rendered chord diagram
    
    Examples
    --------
    >>> # Matrix example
    >>> matrix = np.random.rand(6, 6)
    >>> fig = create_circlize_chord_diagram(matrix, data_type='matrix')
    >>> 
    >>> # Adjacency list example
    >>> df = pd.DataFrame({
    ...     'source': ['A', 'A', 'B', 'C'],
    ...     'target': ['X', 'Y', 'X', 'Z'],
    ...     'value': [5, 3, 4, 6]
    ... })
    >>> fig = create_circlize_chord_diagram(df, data_type='adjacency_list')
    """
    # Update global defaults with user parameters
    global CHORD_DIAGRAM_DEFAULTS
    CHORD_DIAGRAM_DEFAULTS.update({
        'big_gap': big_gap,
        'small_gap': small_gap,
        'track_height': 0.12,
        'grid_alpha': grid_alpha,
        'link_min_width': link_min_width,
        'link_max_width': link_max_width,
        'link_width_scale': link_width_scale,
        'link_alpha': link_alpha,
        'arrow_length': arrow_length,
        'arrow_width': arrow_width,
        'diff_height': diff_height,
        'label_offset': sector_label_offset,
        'label_fontsize': sector_label_fontsize,
        'background_color': background_color,
        'grid_color': grid_color,
        'reduce_threshold': reduce_threshold
    })
    
    # ========================================================================
    # DATA PROCESSING
    # ========================================================================
    links = []
    all_sectors = []
    groups = {}
    
    if data_type == 'matrix':
        # Handle pandas DataFrame or numpy array
        matrix = data.values if isinstance(data, pd.DataFrame) else np.array(data)
        row_names = data.index.tolist() if isinstance(data, pd.DataFrame) else [f"Source_{i+1}" for i in range(matrix.shape[0])]
        col_names = data.columns.tolist() if isinstance(data, pd.DataFrame) else [f"Target_{j+1}" for j in range(matrix.shape[1])]
        
        # Apply symmetry filter if requested
        if symmetric:
            matrix = np.tril(matrix, -1)
        
        # Convert matrix to link list
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if abs(value) > reduce_threshold:
                    links.append({
                        'source': f"S{i+1}",
                        'target': f"T{j+1}",
                        'value': abs(value),
                        'sign': np.sign(value),
                        'original_source': row_names[i],
                        'original_target': col_names[j]
                    })
        
        # Create sector lists with group separation
        row_sectors = [f"S{i+1}" for i in range(matrix.shape[0])]
        col_sectors = [f"T{j+1}" for j in range(matrix.shape[1])]
        all_sectors = row_sectors + col_sectors
        
        # Assign groups (sources=0, targets=1) for visual separation
        groups = {sector: 0 for sector in row_sectors}
        groups.update({sector: 1 for sector in col_sectors})
        
        # Create default sector labels
        if sector_labels is None:
            sector_labels = {}
            for i, name in enumerate(row_names):
                sector_labels[f"S{i+1}"] = name
            for j, name in enumerate(col_names):
                sector_labels[f"T{j+1}"] = name
    
    else:  # adjacency_list
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        
        # Validate DataFrame structure
        if df.shape[1] < 3:
            raise ValueError("Adjacency list requires at least 3 columns: [source, target, value]")
        
        # Build links
        for _, row in df.iterrows():
            source = str(row.iloc[0])
            target = str(row.iloc[1])
            value = float(row.iloc[2]) if len(row) > 2 else 1.0
            
            if abs(value) > reduce_threshold:
                links.append({
                    'source': source,
                    'target': target,
                    'value': abs(value),
                    'sign': np.sign(value) if len(row) > 2 else 1
                })
        
        # Extract unique sectors
        all_sectors = sorted(set(df.iloc[:, 0].astype(str).tolist() + 
                               df.iloc[:, 1].astype(str).tolist()))
        groups = {s: 0 for s in all_sectors}  # Default single group
        
        # Apply custom sector ordering if provided
        if sector_order:
            valid_sectors = [s for s in sector_order if s in all_sectors]
            all_sectors = valid_sectors + [s for s in all_sectors if s not in valid_sectors]
        
        # Create default labels if none provided
        if sector_labels is None:
            sector_labels = {s: s for s in all_sectors}
    
    # ========================================================================
    # DIAGRAM CONSTRUCTION
    # ========================================================================
    diagram = CirclizeChordDiagram(figsize=figsize, dpi=100)
    diagram.big_gap = big_gap
    diagram.small_gap = small_gap
    diagram.set_start_degree(start_degree)
    diagram.set_direction(direction == 'clockwise')
    diagram.background_color = background_color
    
    # Initialize sectors with grouping
    diagram.initialize_sectors(all_sectors, groups)
    
    # Generate sector colors if not provided
    if sector_colors is None:
        sector_colors = {}
        # Use two distinct colormaps for different groups
        cmap_source = plt.cm.tab20
        cmap_target = plt.cm.tab20b
        
        for i, sector in enumerate(all_sectors):
            group_id = groups.get(sector, 0)
            if group_id == 0:
                sector_colors[sector] = cmap_source(i % 20)
            else:
                sector_colors[sector] = cmap_target(i % 20)
    
    # Add tracks to all sectors
    track_height = 0.12
    for sector in all_sectors:
        for track_idx in range(tracks):
            track_color = track_colors[track_idx] if track_colors and track_idx < len(track_colors) else grid_color
            diagram.add_track(
                sector, 
                track_index=track_idx,
                height=track_heights[track_idx] if track_heights and track_idx < len(track_heights) else track_height,
                color=track_color,
                alpha=grid_alpha
            )
    
    # ========================================================================
    # LINK PROCESSING & STYLING
    # ========================================================================
    # Process link colors based on strategy
    link_color_map = {}
    
    if isinstance(link_colors, str):
        if link_colors == 'group':
            for link in links:
                source_group = groups.get(link['source'], 0)
                link_color_map[(link['source'], link['target'])] = sector_colors.get(
                    link['source'] if source_group == 0 else link['target'],
                    '#888888'
                )
        elif link_colors == 'value':
            # Normalize values for colormap
            values = [link['value'] for link in links]
            norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
            cmap = plt.cm.viridis
            for link in links:
                link_color_map[(link['source'], link['target'])] = cmap(norm(link['value']))
        elif link_colors == 'single_color':
            for link in links:
                link_color_map[(link['source'], link['target'])] = '#3B82F6'  # Blue-500
        else:
            for link in links:
                link_color_map[(link['source'], link['target'])] = link_colors
    elif isinstance(link_colors, dict):
        link_color_map = link_colors
    else:
        for link in links:
            link_color_map[(link['source'], link['target'])] = link_colors
    
    # Sort links by value if requested (prevents visual clutter)
    if link_sort:
        links.sort(key=lambda x: x['value'], reverse=link_decreasing)
    
    # Assign z-index based on strategy
    if link_zindex == 'value':
        max_val = max(link['value'] for link in links) if links else 1
        for link in links:
            link['zindex'] = 2 + (link['value'] / max_val) * 8  # Range: 2-10
    elif link_zindex == 'random':
        for link in links:
            link['zindex'] = 2 + random.random() * 8
    elif isinstance(link_zindex, (list, np.ndarray)):
        for i, link in enumerate(links):
            link['zindex'] = link_zindex[i] if i < len(link_zindex) else 5.0
    else:
        for link in links:
            link['zindex'] = 5.0
    
    # Create links with directional features
    for link in links:
        source = link['source']
        target = link['target']
        value = link['value']
        
        # Check for highlighting
        is_highlighted = highlight_links and (source, target) in highlight_links
        
        # Determine color
        color = link_color_map.get((source, target), '#64748B')  # Slate-500 default
        if is_highlighted:
            color = highlight_color
            alpha = highlight_alpha
        else:
            alpha = link_alpha
        
        # Determine directionality
        is_directional = False
        direction_value = 0
        
        if isinstance(directional, dict):
            direction_value = directional.get((source, target), 0)
            is_directional = direction_value != 0
        elif directional and data_type == 'matrix':
            # For matrices, direction is source→target by convention
            is_directional = True
            direction_value = 1
        
        # Create the link
        try:
            diagram.create_link(
                source=source,
                target=target,
                value=value,
                color=color,
                alpha=alpha,
                directional=direction_value if is_directional else False,
                direction_type=direction_type,
                arrow_length=arrow_length,
                arrow_width=arrow_width,
                highlight=is_highlighted,
                zindex=link['zindex']
            )
        except Exception as e:
            st.warning(f"Warning: Could not draw link {source}→{target}: {str(e)}")
            continue
    
    # ========================================================================
    # FINALIZATION
    # ========================================================================
    # Add sector labels
    diagram.add_sector_labels(
        sector_labels,
        fontsize=sector_label_fontsize,
        offset=sector_label_offset
    )
    
    # Finalize and return figure
    fig = diagram.finalize(
        title=title,
        show_frame=show_frame,
        background_color=background_color
    )
    
    return fig


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def create_streamlit_circlize_app():
    """Production-ready Streamlit application for chord diagram creation."""
    
    # Page configuration
    st.set_page_config(
        page_title="🎨 Circlize-Style Chord Diagrams",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    /* Global styles */
    .main {
        padding: 1rem 1.5rem;
    }
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header styling */
    h1 {
        color: #0F172A;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #475569;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Section dividers */
    .section-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #CBD5E1, transparent);
        margin: 1.5rem 0;
    }
    
    /* Cards and containers */
    .stCard {
        background-color: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #E2E8F0;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #2563EB;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F1F5F9 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 36px;
        white-space: nowrap;
        border-radius: 6px 6px 0px 0px;
        background-color: #E2E8F0;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
    
    /* Metrics */
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid #E2E8F0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🎨 Circlize-Style Chord Diagrams")
    st.markdown(
        '<p class="subtitle">Create publication-quality circular visualizations with directional flows, '
        'sector grouping, and advanced styling inspired by R\'s circlize package.</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://via.placeholder.com/150x40/3B82F6/FFFFFF?text=Circlize", 
                use_column_width=True)
        st.markdown("### 📊 Data Configuration")
        
        # Data source selection
        use_example = st.checkbox("✨ Use example dataset", value=True, key="use_example")
        
        if use_example:
            # Generate sophisticated example data
            np.random.seed(42)
            n = 10
            # Create block-structured correlation matrix
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i, j] = 1.0
                    elif abs(i - j) <= 2:
                        matrix[i, j] = 0.7 - abs(i - j) * 0.2
                    else:
                        matrix[i, j] = np.random.uniform(0.1, 0.3)
            
            # Add asymmetry for directional flows
            for i in range(n):
                for j in range(i+1, n):
                    if np.random.random() > 0.7:
                        matrix[i, j] *= 1.5
            
            row_names = [f"Gene_{i+1:02d}" for i in range(n)]
            col_names = [f"Pathway_{j+1:02d}" for j in range(n)]
            data = pd.DataFrame(matrix, index=row_names, columns=col_names)
            data_type = "matrix"
            st.success("✅ Loaded biological interaction example (10×10)")
        else:
            uploaded_file = st.file_uploader("📤 Upload CSV/Excel", type=['csv', 'xlsx'])
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)
                    data_type = st.radio("Data format", ["matrix", "adjacency_list"], 
                                       horizontal=True)
                    if data_type == "adjacency_list":
                        st.info("ℹ️ Expected columns: `source`, `target`, `value`")
                    st.success(f"✅ Loaded {data.shape[0]}×{data.shape[1]} dataset")
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    st.stop()
            else:
                st.info("👆 Upload your data or use the example dataset")
                st.stop()
        
        # Quick actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Presets")
        preset = st.selectbox("Apply preset style", 
                            ["Default", "Flow Analysis", "Correlation Matrix", "Network Graph"])
        if preset == "Flow Analysis":
            st.session_state.update({
                'directional': True,
                'direction_type': ['diffHeight', 'arrows'],
                'link_colors': 'group'
            })
        elif preset == "Correlation Matrix":
            st.session_state.update({
                'symmetric': True,
                'link_colors': 'value',
                'directional': False
            })
        elif preset == "Network Graph":
            st.session_state.update({
                'big_gap': 15,
                'small_gap': 2,
                'link_colors': 'single_color'
            })
    
    # Main content tabs
    tab_viz, tab_data, tab_guide, tab_export = st.tabs([
        "🎨 Visualization", 
        "🔍 Data Explorer", 
        "📘 User Guide", 
        "📤 Export"
    ])
    
    # ============================================================================
    # VISUALIZATION TAB
    # ============================================================================
    with tab_viz:
        st.header("Interactive Chord Diagram")
        
        # Parameter configuration in expanders
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### ⚙️ Rendering Controls")
            
            with st.expander("🧭 Layout & Orientation", expanded=True):
                start_degree = st.slider("Starting angle (°)", 0, 359, 0, 15)
                direction = st.selectbox("Direction", ["clockwise", "counter-clockwise"])
                big_gap = st.slider("Group gap (°)", 0, 30, 10, 1)
                small_gap = st.slider("Sector gap (°)", 0, 10, 2, 1)
            
            with st.expander("🎨 Visual Styling", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    bg_color = st.color_picker("Background", "#FFFFFF")
                    grid_alpha = st.slider("Grid opacity", 0.0, 0.5, 0.12, 0.01)
                with col_b:
                    link_alpha = st.slider("Link opacity", 0.2, 1.0, 0.75, 0.05)
                    show_frame = st.checkbox("Show frame", value=False)
            
            with st.expander("➡️ Directional Features", expanded=False):
                directional = st.checkbox("Enable directional flows", value=False)
                if directional:
                    direction_type = st.multiselect(
                        "Direction indicators",
                        ["diffHeight", "arrows"],
                        default=["diffHeight", "arrows"]
                    )
                    arrow_size = st.slider("Arrow size", 0.5, 2.0, 1.0, 0.1)
            
            with st.expander("⚡ Performance", expanded=False):
                reduce_threshold = st.slider("Min link value", 0.0, 0.5, 0.01, 0.01)
                st.caption("Links below threshold will be hidden")
        
        with col1:
            # Generate diagram with error handling
            try:
                with st.spinner(".Rendering chord diagram..."):
                    # Prepare parameters with session state fallbacks
                    params = {
                        'data': data,
                        'data_type': data_type,
                        'figsize': (14, 14),
                        'title': f"Circlize Diagram • {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        'start_degree': start_degree,
                        'direction': direction,
                        'big_gap': big_gap,
                        'small_gap': small_gap,
                        'link_alpha': link_alpha,
                        'grid_alpha': grid_alpha,
                        'background_color': bg_color,
                        'show_frame': show_frame,
                        'reduce_threshold': reduce_threshold,
                        'directional': directional,
                        'direction_type': direction_type if directional else [],
                        'arrow_length': 0.1 * arrow_size,
                        'arrow_width': 0.05 * arrow_size,
                        'symmetric': (preset == "Correlation Matrix"),
                        'link_colors': 'value' if preset == "Correlation Matrix" else 'group'
                    }
                    
                    # Generate figure
                    fig = create_circlize_chord_diagram(**params)
                    
                    # Display with responsive sizing
                    st.pyplot(fig, use_container_width=True, clear_figure=True)
                    
                    # Store figure in session state for export
                    st.session_state['current_figure'] = fig
                    
            except Exception as e:
                st.error(f"❌ Rendering error: {str(e)}")
                with st.expander("🔍 Show traceback"):
                    st.code(traceback.format_exc())
        
        # Interactive legend
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📌 Diagram Legend")
        col_leg1, col_leg2, col_leg3 = st.columns(3)
        with col_leg1:
            st.markdown("- **Sector width**: Fixed (proportional scaling coming soon)")
        with col_leg2:
            st.markdown("- **Link thickness**: Proportional to connection strength")
        with col_leg3:
            if directional:
                st.markdown("- **Arrow direction**: Flow direction (source → target)")
            else:
                st.markdown("- **Symmetric links**: Undirected relationships")
    
    # ============================================================================
    # DATA EXPLORER TAB
    # ============================================================================
    with tab_data:
        st.header("Dataset Analysis")
        
        if data_type == "matrix":
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            with col_stats1:
                st.metric("Rows (Sources)", data.shape[0])
            with col_stats2:
                st.metric("Columns (Targets)", data.shape[1])
            with col_stats3:
                st.metric("Total Links", f"{np.sum(np.abs(data.values) > reduce_threshold):,}")
            with col_stats4:
                st.metric("Density", f"{np.mean(np.abs(data.values) > reduce_threshold)*100:.1f}%")
            
            st.markdown("#### 🔍 Data Preview")
            st.dataframe(data.style.background_gradient(cmap='RdBu_r', axis=None, vmin=-1, vmax=1),
                        use_container_width=True)
            
            st.markdown("#### 📈 Value Distribution")
            flat_vals = data.values.flatten()
            fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
            ax_hist.hist(flat_vals, bins=50, color='#3B82F6', alpha=0.8, edgecolor='white')
            ax_hist.set_xlabel('Value')
            ax_hist.set_ylabel('Frequency')
            ax_hist.set_title('Distribution of Link Values')
            ax_hist.grid(True, alpha=0.2)
            st.pyplot(fig_hist, use_container_width=True)
        else:
            st.markdown(f"**Links**: {len(data)}")
            st.markdown(f"**Unique Sources**: {data.iloc[:,0].nunique()}")
            st.markdown(f"**Unique Targets**: {data.iloc[:,1].nunique()}")
            st.dataframe(data.head(15), use_container_width=True)
    
    # ============================================================================
    # USER GUIDE TAB
    # ============================================================================
    with tab_guide:
        st.header("📘 Complete User Guide")
        
        st.markdown("""
        ### 🌟 Key Features
        
        #### Directional Flow Visualization
        - **Height Differentiation**: Source links attach lower, targets higher (or vice versa)
        - **Arrows**: Visual indicators showing flow direction
        - **Asymmetric Links**: Represent directed relationships (e.g., gene → pathway regulation)
        
        #### Sector Organization
        - **Group Separation**: Automatic visual grouping with larger gaps between categories
        - **Custom Ordering**: Control sector sequence for optimal pattern visibility
        - **Multi-Track Architecture**: Layer additional data dimensions in concentric rings
        
        #### Advanced Styling
        - **Value-Based Coloring**: Links colored by strength using perceptually uniform colormaps
        - **Group-Based Coloring**: Links inherit color from source or target sector
        - **Highlighting**: Emphasize critical pathways with glow effects
        
        ### 📥 Data Formats
        
        #### Matrix Format (Recommended)
        ```python
        # Rows = sources, Columns = targets
        # Values represent connection strength
        matrix = [
            [0.0, 0.8, 0.3],  # Source A connections
            [0.5, 0.0, 0.9],  # Source B connections
            [0.2, 0.4, 0.0]   # Source C connections
        ]
        ```
        
        #### Adjacency List Format
        ```python
        import pandas as pd
        df = pd.DataFrame({
            'source': ['Gene_A', 'Gene_A', 'Gene_B'],
            'target': ['Pathway_X', 'Pathway_Y', 'Pathway_X'],
            'value': [0.85, 0.42, 0.93]
        })
        ```
        
        ### 💡 Pro Tips
        
        1. **For flow diagrams**: Enable directional links with both height differentiation and arrows
        2. **For correlation matrices**: Use symmetric mode with value-based coloring
        3. **Reduce visual clutter**: Increase the "Min link value" threshold to hide weak connections
        4. **Highlight key relationships**: Use the API's `highlight_links` parameter for emphasis
        5. **Publication quality**: Export as PDF/SVG for vector graphics in papers
        
        ### 🔬 Use Cases
        
        - **Bioinformatics**: Gene-pathway interactions, protein-protein networks
        - **Flow Analysis**: Migration patterns, financial transactions, user journey mapping
        - **Correlation Analysis**: Feature relationships in ML datasets
        - **Comparative Genomics**: Cross-species gene family relationships
        
        ### 🚀 Performance Notes
        
        - Handles datasets up to **500+ links** smoothly in Streamlit
        - For larger networks (>1000 links), consider:
            * Increasing the reduction threshold
            * Pre-filtering weak connections
            * Using server-side rendering
        """)
    
    # ============================================================================
    # EXPORT TAB
    # ============================================================================
    with tab_export:
        st.header("📤 Export Options")
        
        if 'current_figure' not in st.session_state:
            st.info("💡 Create a diagram first, then export it here")
        else:
            fig = st.session_state['current_figure']
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                st.markdown("#### 🖼️ Image Formats")
                dpi = st.slider("Resolution (DPI)", 150, 600, 300, 50)
                
                # PNG Export
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format='png', dpi=dpi, 
                          bbox_inches='tight', facecolor=bg_color)
                buf_png.seek(0)
                st.download_button(
                    label="⬇️ Download PNG (High Quality)",
                    data=buf_png,
                    file_name=f"chord_diagram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    key="png_download"
                )
                
                # SVG Export (vector)
                buf_svg = io.BytesIO()
                fig.savefig(buf_svg, format='svg', bbox_inches='tight')
                buf_svg.seek(0)
                st.download_button(
                    label="⬇️ Download SVG (Editable Vector)",
                    data=buf_svg,
                    file_name=f"chord_diagram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                    mime="image/svg+xml",
                    key="svg_download"
                )
            
            with col_exp2:
                st.markdown("#### 📄 Publication Formats")
                
                # PDF Export
                buf_pdf = io.BytesIO()
                fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
                buf_pdf.seek(0)
                st.download_button(
                    label="⬇️ Download PDF (Print Ready)",
                    data=buf_pdf,
                    file_name=f"chord_diagram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="pdf_download"
                )
                
                # Data export
                if data_type == "matrix":
                    csv_data = data.to_csv(index=True)
                    st.download_button(
                        label="⬇️ Export Data (CSV)",
                        data=csv_data,
                        file_name=f"chord_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="data_download"
                    )
            
            st.markdown("#### 💡 Export Tips")
            st.info(
                """
                - **PNG**: Best for presentations and web (raster format)
                - **SVG**: Ideal for editing in Illustrator/Inkscape (vector format)
                - **PDF**: Required for academic publications (preserves vector quality)
                - For print: Use ≥300 DPI resolution
                """
            )
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Circlize-Style Chord Diagrams • Built with Streamlit & Matplotlib • 
        Inspired by R's circlize package</p>
        <p>💡 Pro Tip: For best results, filter weak connections and use directional indicators 
        for flow visualization</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def matrix_to_adjacency_list(matrix: np.ndarray, 
                           row_names: Optional[List[str]] = None,
                           col_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Convert matrix to adjacency list format."""
    if row_names is None:
        row_names = [f"Row_{i+1}" for i in range(matrix.shape[0])]
    if col_names is None:
        col_names = [f"Col_{j+1}" for j in range(matrix.shape[1])]
    
    records = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if abs(value) > 1e-10:  # Non-zero threshold
                records.append({
                    'source': row_names[i],
                    'target': col_names[j],
                    'value': value
                })
    
    return pd.DataFrame(records)


def adjacency_list_to_matrix(df: pd.DataFrame, 
                           fill_value: float = 0.0) -> Tuple[np.ndarray, List[str], List[str]]:
    """Convert adjacency list to matrix format."""
    sources = sorted(df.iloc[:, 0].unique())
    targets = sorted(df.iloc[:, 1].unique())
    
    matrix = np.full((len(sources), len(targets)), fill_value)
    source_to_idx = {s: i for i, s in enumerate(sources)}
    target_to_idx = {t: j for j, t in enumerate(targets)}
    
    for _, row in df.iterrows():
        i = source_to_idx[row.iloc[0]]
        j = target_to_idx[row.iloc[1]]
        matrix[i, j] = row.iloc[2]
    
    return matrix, sources, targets


def create_comparative_chord_diagrams(data_list: List[Any], 
                                    titles: Optional[List[str]] = None,
                                    ncols: int = 2,
                                    figsize: Tuple[int, int] = (20, 10)) -> plt.Figure:
    """
    Create multiple chord diagrams for comparative analysis.
    
    Note: This is a simplified implementation. For production use,
    consider creating separate diagrams with consistent scaling.
    """
    if titles is None:
        titles = [f"Dataset {i+1}" for i in range(len(data_list))]
    
    nrows = math.ceil(len(data_list) / ncols)
    fig = plt.figure(figsize=figsize)
    
    for idx, (data, title) in enumerate(zip(data_list, titles)):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='polar')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
        
        # Placeholder for actual diagram rendering
        ax.text(0.5, 0.5, 'Diagram Placeholder\n(Feature in development)',
               transform=ax.transAxes,
               ha='center', va='center',
               fontsize=12, color='#64748B')
    
    plt.tight_layout()
    return fig


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Application entry point."""
    create_streamlit_circlize_app()


if __name__ == "__main__":
    main()
