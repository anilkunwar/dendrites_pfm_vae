# -*- coding: utf-8 -*-
"""
Enhanced Vivid Chord Diagram Visualization with Large Canvas and High Resolution
Ultra-high resolution circular network visualizations with vivid colors and glow effects
Optimized for publications, presentations, and large-format displays
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch, FancyBboxPatch, Wedge, Arc, Circle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
import matplotlib.patheffects as patheffects
import colorsys
from scipy import stats
import io
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import math
import random
from datetime import datetime
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FuncFormatter
import traceback
import os
import sys

warnings.filterwarnings('ignore')

# ============================================================================
# ENHANCED CONFIGURATION & CONSTANTS - LARGER & MORE VIVID
# ============================================================================

CHORD_DIAGRAM_DEFAULTS = {
    # LARGER CANVAS
    'figsize': (24, 24),           # Increased from (14, 14) to (24, 24)
    'dpi': 300,                    # Increased from 100 to 300 for high quality
    'big_gap': 15.0,               # degrees between groups
    'small_gap': 2.0,              # degrees within groups
    'start_degree': 0,
    'clockwise': True,
    
    # ENHANCED TRACK DIMENSIONS
    'track_height': 0.20,          # Increased from 0.12 to 0.20
    'track_padding': 0.03,
    'grid_alpha': 0.10,
    
    # VIVID LINK STYLING
    'link_min_width': 1.5,         # Increased from 0.8
    'link_max_width': 25.0,        # Increased from 10.0
    'link_width_scale': 5.0,       # Increased from 3.5
    'link_alpha': 0.85,            # Increased from 0.75
    'link_glow_intensity': 2.5,    # NEW: Glow effect multiplier
    'link_glow_alpha': 0.4,        # NEW: Glow transparency
    'link_border_width': 0.8,      # NEW: Border around links
    'link_border_color': 'white',  # NEW: Border color
    
    # ENHANCED DIRECTIONAL FEATURES
    'arrow_length': 0.18,          # Increased from 0.12
    'arrow_width': 0.10,           # Increased from 0.06
    'diff_height': 0.06,           # Increased from 0.03
    
    # IMPROVED LABELING
    'label_offset': 0.30,          # Increased from 0.18
    'label_fontsize': 18,          # Increased from 13
    'label_fontweight': 'bold',
    'label_shadow': True,
    'label_bg_alpha': 0.95,
    
    # VIVID COLORS & BACKGROUNDS
    'background_color': '#0A0F1A', # Dark navy for contrast
    'grid_color': '#2A3B5C',
    'highlight_alpha': 0.98,
    'highlight_glow': 3.0,
    
    # NETWORK ENHANCEMENTS
    'reduce_threshold': 0.001,     # Lower threshold to show more links
    'node_size_min': 150,          # NEW: Minimum node size
    'node_size_max': 800,          # NEW: Maximum node size
    'show_node_borders': True,
    'node_border_width': 3,
    'sector_glow': True,
    'sector_glow_width': 8,
    'sector_glow_alpha': 0.3,
    
    # ADVANCED EFFECTS
    'radial_gradient': True,
    'show_legend': True,
    'legend_fontsize': 14,
    'colorbar_fontsize': 12,
    'show_statistics': True,
    'statistics_fontsize': 12,
    
    # PERFORMANCE
    'max_links': 1000,             # Limit for very large networks
    'simplify_curves': False       # Keep detailed curves
}

# ============================================================================
# VIVID COLOR PALETTES
# ============================================================================

class VividColorPalettes:
    """Collection of vivid, high-contrast color palettes for network visualization"""
    
    @staticmethod
    def create_gradient_cmap(colors, name='custom_gradient'):
        """Create smooth gradient colormap from color list"""
        return LinearSegmentedColormap.from_list(name, colors, N=256)
    
    @staticmethod
    def get_vivid_palette(n_colors=20):
        """Get vivid, distinguishable colors using seaborn"""
        return sns.color_palette("husl", n_colors)
    
    @staticmethod
    def get_fire_palette():
        """Hot fire colors: black -> red -> orange -> yellow -> white"""
        return VividColorPalettes.create_gradient_cmap([
            '#000000', '#4B0000', '#8B0000', '#FF0000', 
            '#FF4500', '#FF8C00', '#FFD700', '#FFFFFF'
        ], 'fire')
    
    @staticmethod
    def get_ocean_palette():
        """Ocean colors: dark blue -> cyan -> white"""
        return VividColorPalettes.create_gradient_cmap([
            '#000033', '#000066', '#003399', '#0066CC', 
            '#0099CC', '#00CCCC', '#66FFFF', '#FFFFFF'
        ], 'ocean')
    
    @staticmethod
    def get_rainbow_palette():
        """Vivid rainbow colors"""
        return VividColorPalettes.create_gradient_cmap([
            '#FF0000', '#FF7F00', '#FFFF00', '#7FFF00',
            '#00FF00', '#00FF7F', '#00FFFF', '#007FFF',
            '#0000FF', '#7F00FF', '#FF00FF', '#FF007F'
        ], 'rainbow')
    
    @staticmethod
    def get_electric_palette():
        """Electric/neon colors"""
        return VividColorPalettes.create_gradient_cmap([
            '#000000', '#1E90FF', '#00FFFF', '#7FFF00',
            '#FFFF00', '#FFA500', '#FF0000', '#FF00FF', '#FFFFFF'
        ], 'electric')
    
    @staticmethod
    def get_matrix_palette():
        """Matrix-style green gradient"""
        return VividColorPalettes.create_gradient_cmap([
            '#000000', '#001100', '#002200', '#003300',
            '#005500', '#008800', '#00AA00', '#00FF00'
        ], 'matrix')
    
    @staticmethod
    def get_purple_gold_palette():
        """Royal purple to gold gradient"""
        return VividColorPalettes.create_gradient_cmap([
            '#1A0033', '#2D004D', '#400066', '#530080',
            '#660099', '#8000CC', '#9933FF', '#B366FF',
            '#CC99FF', '#E6CCFF', '#FFFFE6', '#FFFFCC',
            '#FFFF99', '#FFFF66', '#FFFF33', '#FFFF00'
        ], 'purple_gold')
    
    @staticmethod
    def get_sector_colors(n_sectors, palette='vivid'):
        """Generate sector colors based on palette choice"""
        if palette == 'vivid':
            colors = VividColorPalettes.get_vivid_palette(n_sectors)
        elif palette == 'fire':
            cmap = VividColorPalettes.get_fire_palette()
            colors = [cmap(i / n_sectors) for i in range(n_sectors)]
        elif palette == 'ocean':
            cmap = VividColorPalettes.get_ocean_palette()
            colors = [cmap(i / n_sectors) for i in range(n_sectors)]
        elif palette == 'rainbow':
            cmap = VividColorPalettes.get_rainbow_palette()
            colors = [cmap(i / n_sectors) for i in range(n_sectors)]
        elif palette == 'electric':
            cmap = VividColorPalettes.get_electric_palette()
            colors = [cmap(i / n_sectors) for i in range(n_sectors)]
        elif palette == 'matrix':
            cmap = VividColorPalettes.get_matrix_palette()
            colors = [cmap(i / n_sectors) for i in range(n_sectors)]
        else:  # Default tab20
            cmap = plt.cm.tab20
            colors = [cmap(i % 20) for i in range(n_sectors)]
        
        return colors


# ============================================================================
# ENHANCED CHORD DIAGRAM ENGINE - LARGER & MORE VIVID
# ============================================================================

class VividChordDiagram:
    """
    Enhanced chord diagram renderer with larger canvas and vivid visualization.
    
    Features:
        - Ultra-large canvas (24x24 inches) at 300 DPI
        - Vivid color schemes with high contrast
        - Glow effects for links and sectors
        - Enhanced directional indicators
        - Node sizing based on connectivity
        - Radial gradients and visual effects
        - Comprehensive legends and statistics
    """
    
    def __init__(self, figsize: Tuple[int, int] = (24, 24), dpi: int = 300,
                 background_color: str = '#0A0F1A'):
        """
        Initialize enhanced chord diagram canvas.
        
        Parameters
        ----------
        figsize : tuple
            Figure dimensions (width, height) in inches - LARGER for vivid display
        dpi : int
            Resolution in dots per inch - HIGHER for crisp rendering
        background_color : str
            Canvas background color - DARK for contrast
        """
        self.figsize = figsize
        self.dpi = dpi
        self.background_color = background_color
        
        # Create figure with larger dimensions
        self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=background_color)
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_theta_zero_location("N")  # 0° at top
        self.ax.set_theta_direction(-1)       # Clockwise by default
        
        # Data structures - ✅ FIXED: Removed space from variable name
        self.sectors: List[str] = []
        self.sector_dict: Dict[str, Dict] = {}  # ✅ CORRECTED: Removed space
        self.links: List[Dict] = []
        self.tracks: Dict[str, Dict[int, Dict]] = {}
        self.groups: Dict[str, int] = {}
        self.node_degrees: Dict[str, float] = {}  # For node sizing
        
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
        self.link_glow_intensity: float = CHORD_DIAGRAM_DEFAULTS['link_glow_intensity']
        self.sector_glow: bool = CHORD_DIAGRAM_DEFAULTS['sector_glow']
        
        # Statistics tracking
        self.stats = {
            'total_links': 0,
            'total_value': 0.0,
            'max_link_value': 0.0,
            'avg_link_value': 0.0,
            'link_density': 0.0
        }
    
    def initialize_sectors(self, sectors: List[str], groups: Optional[Dict[str, int]] = None,
                          sector_sizes: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize sectors with optional grouping and sizing.
        
        Parameters
        ----------
        sectors : list of str
            List of sector identifiers in display order
        groups : dict, optional
            Mapping from sector name to group ID (creates larger gaps between groups)
        sector_sizes : dict, optional
            Custom sizes for sectors (normalized to sum to 1)
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
        
        # Initialize node degrees for sizing
        self.node_degrees = {sector: 0.0 for sector in sectors}
    
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
                'mid': (current_angle + sector_width / 2) % 360,
                'width': sector_width
            }
            current_angle += sector_width + self.gap_after.get(sector, self.small_gap)
        
        self.sector_angles = angles
        return angles
    
    def draw_sector_track(self, sector: str, angles: Dict[str, float], 
                         color: Union[str, Tuple] = '#2A3B5C', 
                         alpha: float = 0.15,
                         track_height: float = 0.20, 
                         track_index: int = 0,
                         glow: bool = True,
                         border_color: Optional[str] = None,
                         border_width: float = 2.0) -> Dict[str, float]:
        """
        Render a single sector track with enhanced visual effects.
        
        Parameters
        ----------
        sector : str
            Sector identifier
        angles : dict
            Angular boundaries with 'start', 'end', 'mid' keys (degrees)
        color : str or RGB tuple
            Fill color for the track
        alpha : float
            Transparency level [0-1]
        track_height : float
            Radial height of the track
        track_index : int
            Track position (0 = innermost)
        glow : bool
            Add glow effect around sector
        border_color : str, optional
            Color for sector border
        border_width : float
            Width of sector border
        
        Returns
        -------
        dict
            Radial boundaries {'inner': r_inner, 'outer': r_outer, 'mid_angle': angle}
        """
        start_rad = np.radians(angles['start'])
        end_rad = np.radians(angles['end'])
        
        # Handle sectors crossing 0° boundary
        if end_rad < start_rad and abs(end_rad - start_rad) < np.pi:
            end_rad += 2 * np.pi
        
        # Create smooth arc
        theta = np.linspace(start_rad, end_rad, 200)  # More points for smoothness
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
        
        # Create and add main track patch
        poly = plt.Polygon(vertices, facecolor=color, alpha=alpha, 
                          edgecolor='none', zorder=0.5)
        self.ax.add_patch(poly)
        
        # Add glow effect
        if glow and CHORD_DIAGRAM_DEFAULTS['sector_glow']:
            glow_width = CHORD_DIAGRAM_DEFAULTS['sector_glow_width']
            glow_alpha = CHORD_DIAGRAM_DEFAULTS['sector_glow_alpha']
            
            # Outer glow
            theta_glow = np.linspace(start_rad, end_rad, 150)
            r_glow_inner = r_outer
            r_glow_outer = r_outer + glow_width / 100.0
            
            theta_glow_poly = np.concatenate([theta_glow, theta_glow[::-1]])
            r_glow_poly = np.concatenate([np.full_like(theta_glow, r_glow_inner),
                                         np.full_like(theta_glow, r_glow_outer)])
            
            x_glow = r_glow_poly * np.cos(theta_glow_poly)
            y_glow = r_glow_poly * np.sin(theta_glow_poly)
            vertices_glow = np.column_stack([x_glow, y_glow])
            
            glow_color = color if isinstance(color, str) else mcolors.to_hex(color)
            glow_poly = plt.Polygon(vertices_glow, facecolor=glow_color, 
                                   alpha=glow_alpha * 0.6, edgecolor='none', zorder=0.4)
            self.ax.add_patch(glow_poly)
        
        # Add border if requested
        if border_color and CHORD_DIAGRAM_DEFAULTS['show_node_borders']:
            border_alpha = min(1.0, alpha * 1.5)
            border_poly = plt.Polygon(vertices, facecolor='none', 
                                     edgecolor=border_color, alpha=border_alpha,
                                     linewidth=border_width, zorder=1.5)
            self.ax.add_patch(border_poly)
        
        return {'inner': r_inner, 'outer': r_outer, 'mid_angle': np.radians(angles['mid'])}
    
    def create_vivid_link(self, source: str, target: str, value: float,
                         source_track: int = 0, target_track: int = 0,
                         color: Union[str, Tuple] = '#FF6B6B', 
                         alpha: float = 0.85,
                         directional: Union[bool, int] = False, 
                         direction_type: Union[str, List[str]] = 'diffHeight',
                         arrow_length: float = 0.18, 
                         arrow_width: float = 0.10,
                         highlight: bool = False, 
                         zindex: float = 1.0,
                         add_glow: bool = True,
                         add_border: bool = True,
                         gradient: bool = False) -> Dict[str, Any]:
        """
        Create a vivid curved link (chord) between two sectors with enhanced styling.
        
        Parameters
        ----------
        source, target : str
            Source and target sector identifiers
        value : float
            Link weight (controls visual width)
        source_track, target_track : int
            Track indices for connection points
        color : str or RGB tuple
            Link color - VIVID colors for visibility
        alpha : float
            Transparency [0-1]
        directional : bool or int
            Direction indicator: False=undirected, 1=source→target, -1=target→source
        direction_type : str or list
            Visual direction cues: 'diffHeight', 'arrows', or both
        arrow_length, arrow_width : float
            Arrow dimensions (normalized units) - LARGER for visibility
        highlight : bool
            Apply visual emphasis (intense glow effect)
        zindex : float
            Rendering order (higher = on top)
        add_glow : bool
            Add glow effect around link
        add_border : bool
            Add white border around link for contrast
        gradient : bool
            Apply color gradient along the link
        
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
        source_r = self.tracks.get(source, {}).get(source_track, {}).get('outer', 0.20)
        target_r = self.tracks.get(target, {}).get(target_track, {}).get('outer', 0.20)
        
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
        
        # Create quadratic Bezier curve in polar space with more control
        control_angle = (source_rad + target_rad) / 2
        control_r = max(source_r, target_r) * 1.6  # Increased bulge for visibility
        
        # Parameterize curve with more points for smoothness
        t = np.linspace(0, 1, 100)  # Increased from 60 to 100
        theta_curve = (1 - t)**2 * source_rad + 2 * (1 - t) * t * control_angle + t**2 * target_rad
        r_curve = (1 - t)**2 * source_r + 2 * (1 - t) * t * control_r + t**2 * target_r
        
        # Normalize angles to [0, 2π)
        theta_curve = theta_curve % (2 * np.pi)
        
        # Calculate base linewidth
        base_width = max(CHORD_DIAGRAM_DEFAULTS['link_min_width'], 
                        min(CHORD_DIAGRAM_DEFAULTS['link_max_width'], 
                            value * CHORD_DIAGRAM_DEFAULTS['link_width_scale']))
        
        # Add glow effect FIRST (behind main link)
        if add_glow and CHORD_DIAGRAM_DEFAULTS['link_glow_intensity'] > 0:
            glow_width = base_width * CHORD_DIAGRAM_DEFAULTS['link_glow_intensity']
            glow_alpha = CHORD_DIAGRAM_DEFAULTS['link_glow_alpha']
            
            # Multi-layer glow for vivid effect
            for i, (w_mult, a_mult) in enumerate([(2.5, 0.2), (1.8, 0.3), (1.2, 0.4)]):
                glow_line, = self.ax.plot(theta_curve, r_curve,
                                        color=color, 
                                        alpha=glow_alpha * a_mult * (alpha if not highlight else 1.0),
                                        linewidth=glow_width * w_mult,
                                        solid_capstyle='round',
                                        solid_joinstyle='round',
                                        zorder=zindex - 0.5 + i * 0.1)
        
        # Add border for contrast
        if add_border and CHORD_DIAGRAM_DEFAULTS['link_border_width'] > 0:
            border_line, = self.ax.plot(theta_curve, r_curve,
                                      color=CHORD_DIAGRAM_DEFAULTS['link_border_color'],
                                      alpha=min(1.0, alpha * 1.2),
                                      linewidth=base_width + CHORD_DIAGRAM_DEFAULTS['link_border_width'] * 2,
                                      solid_capstyle='round',
                                      solid_joinstyle='round',
                                      zorder=zindex - 0.1)
        
        # Create gradient-colored link if requested
        if gradient and isinstance(color, tuple) and len(color) == 2:
            # Create LineCollection with gradient
            points = np.array([theta_curve, r_curve]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            lc = LineCollection(segments,
                              colors=[color[0] if i < len(segments)//2 else color[1] 
                                     for i in range(len(segments))],
                              linewidths=base_width,
                              alpha=alpha,
                              capstyle='round',
                              zorder=zindex)
            self.ax.add_collection(lc)
            line = lc
        else:
            # Render base link
            line, = self.ax.plot(theta_curve, r_curve,
                               color=color, 
                               alpha=alpha,
                               linewidth=base_width,
                               solid_capstyle='round',
                               solid_joinstyle='round',
                               zorder=zindex)
        
        # Add directional arrows if requested
        if directional and ('arrows' in direction_type or direction_type == 'arrows'):
            # Position arrows at multiple points for better visibility
            arrow_positions = [0.4, 0.6, 0.8] if value > 0.5 else [0.6]  # More arrows for strong links
            
            for arrow_pos in arrow_positions:
                arrow_idx = int(len(t) * arrow_pos)
                arrow_theta = theta_curve[arrow_idx]
                arrow_r = r_curve[arrow_idx]
                
                # Compute tangent direction for arrow orientation
                if arrow_idx > 2 and arrow_idx < len(t) - 2:
                    dx = r_curve[arrow_idx + 2] * np.cos(theta_curve[arrow_idx + 2]) - \
                         r_curve[arrow_idx - 2] * np.cos(theta_curve[arrow_idx - 2])
                    dy = r_curve[arrow_idx + 2] * np.sin(theta_curve[arrow_idx + 2]) - \
                         r_curve[arrow_idx - 2] * np.sin(theta_curve[arrow_idx - 2])
                    
                    # Create arrow using FancyArrowPatch in Cartesian space
                    x = arrow_r * np.cos(arrow_theta)
                    y = arrow_r * np.sin(arrow_theta)
                    
                    arrow = FancyArrowPatch(
                        (x - dx/15, y - dy/15),
                        (x + dx/15, y + dy/15),
                        arrowstyle=f'->,head_width={arrow_width*12},head_length={arrow_length*8}',
                        color=color,
                        alpha=min(1.0, alpha * 1.3),  # More opaque arrows
                        linewidth=base_width * 0.6,
                        mutation_scale=25,
                        zorder=zindex + 0.5
                    )
                    self.ax.add_patch(arrow)
        
        # Apply highlight effect (intense glow)
        if highlight:
            highlight_glow = CHORD_DIAGRAM_DEFAULTS['highlight_glow']
            # White glow
            self.ax.plot(theta_curve, r_curve,
                       color='white', alpha=0.6,
                       linewidth=base_width * highlight_glow * 1.2,
                       solid_capstyle='round',
                       zorder=zindex - 0.3)
            # Color glow
            self.ax.plot(theta_curve, r_curve,
                       color=color, alpha=0.8,
                       linewidth=base_width * highlight_glow,
                       solid_capstyle='round',
                       zorder=zindex - 0.2)
            # Enhanced main line
            self.ax.plot(theta_curve, r_curve,
                       color=color, alpha=min(1.0, alpha * 1.4),
                       linewidth=base_width * 1.3,
                       solid_capstyle='round',
                       zorder=zindex + 0.3)
        
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
            'zindex': zindex,
            'width': base_width
        }
        self.links.append(link_data)
        
        # Update node degrees for sizing
        self.node_degrees[source] = self.node_degrees.get(source, 0.0) + value
        self.node_degrees[target] = self.node_degrees.get(target, 0.0) + value
        
        return link_data
    
    def add_track(self, sector: str, track_index: int = 0, height: Optional[float] = None,
                 color: Union[str, Tuple] = '#2A3B5C', 
                 alpha: Optional[float] = None,
                 glow: bool = True,
                 border_color: Optional[str] = None) -> Dict[str, float]:
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
        color : str or tuple
            Track fill color
        alpha : float, optional
            Transparency (uses default if None)
        glow : bool
            Add glow effect
        border_color : str, optional
            Border color for track
        
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
        
        # Determine border color if not specified
        if border_color is None and isinstance(color, str):
            # Lighten the color for border
            border_color = self._lighten_color(color, 0.3)
        
        track_data = self.draw_sector_track(
            sector, 
            self.sector_angles[sector],
            color=color,
            alpha=alpha,
            track_height=height,
            track_index=track_index,
            glow=glow,
            border_color=border_color,
            border_width=CHORD_DIAGRAM_DEFAULTS['node_border_width']
        )
        self.tracks[sector][track_index] = track_data
        return track_data
    
    def add_sector_labels(self, label_dict: Optional[Dict[str, str]] = None, 
                         fontsize: Optional[int] = None,
                         offset: Optional[float] = None, 
                         rotation: str = 'auto',
                         fontweight: str = 'bold',
                         fontfamily: str = 'sans-serif',
                         shadow: bool = True,
                         bg_color: str = 'white',
                         bg_alpha: float = 0.95) -> None:
        """
        Add vivid, readable labels to sectors with enhanced styling.
        
        Parameters
        ----------
        label_dict : dict, optional
            Custom label mapping {sector: label_text}
        fontsize : int, optional
            Font size (uses default if None) - LARGER for visibility
        offset : float, optional
            Radial distance from outer track edge
        rotation : str
            'auto' (follows circle), 'horizontal', or fixed angle
        fontweight : str
            Font weight ('normal', 'bold', etc.)
        fontfamily : str
            Font family name
        shadow : bool
            Add text shadow for readability
        bg_color : str
            Background color for label box
        bg_alpha : float
            Background transparency
        """
        if not self.sector_angles:
            self.compute_sector_angles()
        
        if fontsize is None:
            fontsize = CHORD_DIAGRAM_DEFAULTS['label_fontsize']
        if offset is None:
            offset = CHORD_DIAGRAM_DEFAULTS['label_offset']
        if fontweight is None:
            fontweight = CHORD_DIAGRAM_DEFAULTS['label_fontweight']
        if shadow is None:
            shadow = CHORD_DIAGRAM_DEFAULTS['label_shadow']
        if bg_alpha is None:
            bg_alpha = CHORD_DIAGRAM_DEFAULTS['label_bg_alpha']
        
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
            
            # Determine text color based on background
            text_color = '#FFFFFF' if self.background_color in ['#000000', '#0A0F1A', '#1A1A1A'] else '#2D3748'
            
            # Add label with enhanced styling
            txt = self.ax.text(
                angle_rad, label_r, label_text,
                fontsize=fontsize, 
                fontweight=fontweight,
                fontfamily=fontfamily,
                rotation=rotation_deg, 
                rotation_mode='anchor',
                ha=ha, 
                va='center',
                zorder=1000,
                color=text_color,
                alpha=0.98
            )
            
            # Add text shadow for better readability
            if shadow:
                txt.set_path_effects([
                    patheffects.Stroke(linewidth=3, foreground='black', alpha=0.7),
                    patheffects.Normal()
                ])
            
            # Add background box with rounded corners
            txt.set_bbox(dict(
                boxstyle="round,pad=0.4",
                facecolor=bg_color,
                edgecolor='none',
                alpha=bg_alpha,
                linewidth=2
            ))
    
    def add_sector_nodes(self, sector_colors: Dict[str, str],
                        node_size_scale: float = 1.0,
                        show_labels: bool = True) -> None:
        """
        Add decorative nodes/circles at sector positions for visual enhancement.
        
        Parameters
        ----------
        sector_colors : dict
            Color mapping for each sector
        node_size_scale : float
            Scale factor for node sizes
        show_labels : bool
            Show sector names on nodes
        
        ✅ FIXED: Properly positioned nodes using Cartesian coordinates
        ✅ FIXED: Removed duplicate radius specification in Circle constructor
        """
        if not self.sector_angles:
            self.compute_sector_angles()
        
        for sector, angle_data in self.sector_angles.items():
            angle_rad = np.radians(angle_data['mid'])
            
            # Get outer radius
            outer_r = 0.0
            if sector in self.tracks:
                for track_data in self.tracks[sector].values():
                    outer_r = max(outer_r, track_data['outer'])
            
            # Calculate node size based on connectivity
            base_size = CHORD_DIAGRAM_DEFAULTS['node_size_min']
            max_size = CHORD_DIAGRAM_DEFAULTS['node_size_max']
            
            degree = self.node_degrees.get(sector, 0.0)
            max_degree = max(self.node_degrees.values()) if self.node_degrees else 1.0
            
            if max_degree > 0:
                size_factor = (degree / max_degree) ** 0.5  # Square root for better scaling
                node_size = base_size + (max_size - base_size) * size_factor
            else:
                node_size = base_size
            
            node_size *= node_size_scale
            
            # ✅ FIXED: Calculate proper Cartesian position for node
            x_pos = outer_r * np.cos(angle_rad)
            y_pos = outer_r * np.sin(angle_rad)
            
            # Calculate appropriate radii for glow and node (scaled for 24x24 canvas)
            glow_radius = node_size / 3000  # Scaled for large canvas
            node_radius = node_size / 2000  # Scaled for large canvas
            
            # Create outer glow circle - ✅ FIXED RADIUS CONFLICT
            glow_circle = Circle(
                (x_pos, y_pos),           # Center position (Cartesian)
                radius=glow_radius,       # Single radius specification
                facecolor=sector_colors.get(sector, '#666666'),
                alpha=0.3,
                zorder=5
            )
            self.ax.add_patch(glow_circle)
            
            # Create main node circle - ✅ FIXED RADIUS CONFLICT
            node_color = sector_colors.get(sector, '#666666')
            node_circle = Circle(
                (x_pos, y_pos),           # Center position (Cartesian)
                radius=node_radius,       # Single radius specification
                facecolor=node_color,
                edgecolor='white' if CHORD_DIAGRAM_DEFAULTS['show_node_borders'] else node_color,
                linewidth=CHORD_DIAGRAM_DEFAULTS['node_border_width'],
                alpha=0.95,
                zorder=10
            )
            self.ax.add_patch(node_circle)
    
    def add_legend(self, title: str = "Link Strength", 
                  position: str = 'right',
                  cmap: Any = None,
                  vmin: float = 0,
                  vmax: float = 1,
                  fontsize: int = 14) -> None:
        """
        Add color legend/bar to the diagram.
        
        Parameters
        ----------
        title : str
            Legend title
        position : str
            Position of legend ('right', 'left', 'bottom')
        cmap : colormap
            Colormap to display
        vmin, vmax : float
            Value range for colormap
        fontsize : int
            Font size for legend text
        """
        if cmap is None:
            cmap = VividColorPalettes.get_rainbow_palette()
        
        # Create inset axes for colorbar
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        if position == 'right':
            axins = inset_axes(self.ax, width="5%", height="50%", 
                             loc='center right', borderpad=3)
        elif position == 'left':
            axins = inset_axes(self.ax, width="5%", height="50%",
                             loc='center left', borderpad=3)
        else:  # bottom
            axins = inset_axes(self.ax, width="50%", height="5%",
                             loc='lower center', borderpad=3)
        
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cb = ColorbarBase(axins, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_label(title, fontsize=fontsize, fontweight='bold', color='white')
        cb.ax.tick_params(labelsize=fontsize-2, colors='white')
        
        # Set background for colorbar
        cb.ax.set_facecolor('none')
    
    def add_statistics_box(self, stats: Dict[str, Any],
                          position: Tuple[float, float] = (0.02, 0.02),
                          fontsize: int = 12) -> None:
        """
        Add statistics text box to the diagram.
        
        Parameters
        ----------
        stats : dict
            Statistics to display
        position : tuple
            Position in figure coordinates (x, y)
        fontsize : int
            Font size for statistics text
        """
        stats_text = []
        for key, value in stats.items():
            if isinstance(value, float):
                stats_text.append(f"{key}: {value:.2f}")
            else:
                stats_text.append(f"{key}: {value}")
        
        stats_str = "\n".join(stats_text)
        
        # Add text box
        self.fig.text(position[0], position[1], stats_str,
                     fontsize=fontsize,
                     fontfamily='monospace',
                     fontweight='bold',
                     color='white',
                     alpha=0.95,
                     bbox=dict(boxstyle='round,pad=0.8',
                             facecolor='#1A2B4A',
                             edgecolor='#4A6B9A',
                             linewidth=2,
                             alpha=0.85),
                     transform=self.fig.transFigure,
                     verticalalignment='bottom',
                     horizontalalignment='left',
                     zorder=10000)
    
    def finalize(self, title: str = "", show_frame: bool = False, 
                background_color: Optional[str] = None,
                show_grid: bool = False) -> plt.Figure:
        """
        Finalize diagram appearance and layout.
        
        Parameters
        ----------
        title : str
            Diagram title - ENHANCED styling
        show_frame : bool
            Display polar coordinate frame
        background_color : str, optional
            Canvas background color
        show_grid : bool
            Show radial grid lines
        
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
        
        self.ax.set_ylim(0, max_radius * 1.4)  # Extra space for labels and decorations
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        
        # Frame visibility
        self.ax.spines['polar'].set_visible(show_frame)
        if show_frame:
            self.ax.spines['polar'].set_color('#666666')
            self.ax.spines['polar'].set_linewidth(2.0)
        
        # Add grid if requested
        if show_grid:
            self.ax.grid(True, alpha=0.2, color='#444444', linewidth=1.0)
        
        # Add title with enhanced styling
        if title:
            # Add title with shadow and glow
            title_text = self.ax.set_title(
                title, 
                fontsize=28,  # LARGER title
                fontweight='bold', 
                pad=40,
                color='white',
                fontfamily='sans-serif',
                alpha=0.98
            )
            
            # Add shadow effect
            title_text.set_path_effects([
                patheffects.Stroke(linewidth=4, foreground='black', alpha=0.8),
                patheffects.Normal()
            ])
        
        # Remove margins
        self.fig.tight_layout(pad=2.0)
        return self.fig
    
    def _lighten_color(self, color: str, amount: float = 0.5) -> str:
        """
        Lighten a color by a given amount.
        
        Parameters
        ----------
        color : str
            Color string (hex or name)
        amount : float
            Amount to lighten (0-1)
        
        Returns
        -------
        str
            Lightened color hex string
        """
        try:
            c = mcolors.to_rgb(color)
            c = colorsys.rgb_to_hls(*c)
            new_color = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
            return mcolors.to_hex(new_color)
        except:
            return '#FFFFFF'


# ============================================================================
# ENHANCED HIGH-LEVEL API - VIVID NETWORK VISUALIZATION
# ============================================================================

def create_vivid_chord_diagram(
    data: Any,  # ✅ FIXED: Added parameter name
    data_type: str = 'matrix',
    figsize: Tuple[int, int] = (24, 24),  # LARGER canvas
    dpi: int = 300,                       # HIGHER resolution
    title: str = "Vivid Network Chord Diagram",
    
    # Layout parameters
    start_degree: float = 0,
    direction: str = 'clockwise',
    big_gap: float = 15.0,
    small_gap: float = 2.0,
    sector_order: Optional[List[str]] = None,
    
    # Sector styling - ENHANCED
    sector_colors: Optional[Union[str, Dict[str, str]]] = 'vivid',
    sector_color_palette: str = 'vivid',  # 'vivid', 'fire', 'ocean', 'rainbow', etc.
    sector_labels: Optional[Dict[str, str]] = None,
    sector_label_fontsize: int = 18,      # LARGER labels
    sector_label_fontweight: str = 'bold',
    sector_label_offset: float = 0.30,    # FURTHER from center
    sector_label_shadow: bool = True,
    show_sector_nodes: bool = True,
    sector_node_scale: float = 1.2,
    
    # Link styling - VIVID
    link_colors: Union[str, Dict[Tuple[str, str], str]] = 'gradient',
    link_color_palette: str = 'rainbow',  # Colormap for value-based coloring
    link_alpha: float = 0.85,
    link_width_scale: float = 5.0,        # LARGER scale
    link_min_width: float = 1.5,          # WIDER minimum
    link_max_width: float = 25.0,         # WIDER maximum
    link_glow: bool = True,
    link_glow_intensity: float = 2.5,
    link_border: bool = True,
    link_border_width: float = 0.8,
    link_border_color: str = 'white',
    
    # Directional features - ENHANCED
    directional: Union[bool, Dict[Tuple[str, str], int]] = False,
    direction_type: Union[str, List[str]] = ['diffHeight', 'arrows'],
    arrow_length: float = 0.18,           # LARGER arrows
    arrow_width: float = 0.10,            # WIDER arrows
    diff_height: float = 0.06,            # MORE height difference
    
    # Highlighting
    highlight_links: Optional[List[Tuple[str, str]]] = None,
    highlight_color: str = '#FF4444',     # Vivid red
    highlight_alpha: float = 0.98,
    
    # Scaling
    scale: bool = False,
    scale_mode: str = 'absolute',
    
    # Advanced features
    symmetric: bool = False,
    reduce_threshold: float = 0.001,      # LOWER threshold to show more
    link_sort: bool = True,
    link_decreasing: bool = True,
    link_zindex: Union[str, List[float]] = 'value',
    link_gradient: bool = False,          # NEW: Gradient along links
    
    # Visual effects - VIVID
    background_color: str = '#0A0F1A',    # Dark for contrast
    grid_color: str = '#2A3B5C',
    grid_alpha: float = 0.10,
    sector_glow: bool = True,
    sector_glow_width: float = 8,
    sector_glow_alpha: float = 0.3,
    show_frame: bool = False,
    show_grid: bool = False,
    radial_gradient: bool = False,
    
    # Multiple tracks - ENHANCED
    tracks: int = 1,
    track_heights: Optional[List[float]] = None,
    track_colors: Optional[List[str]] = None,
    track_alpha: float = 0.15,
    
    # Legends and statistics
    show_legend: bool = True,
    legend_title: str = "Connection Strength",
    legend_position: str = 'right',
    show_statistics: bool = True,
    statistics_position: Tuple[float, float] = (0.02, 0.02),
    statistics_fontsize: int = 12,
    
    # Performance
    max_links: int = 1000,
    simplify_curves: bool = False
) -> plt.Figure:
    """
    Create vivid, large-scale chord diagrams with enhanced network visualization.
    
    This function creates publication-quality chord diagrams optimized for
    vivid display of complex networks with enhanced colors, glow effects,
    and visual hierarchy.
    
    Parameters
    ----------
    data : array-like or DataFrame
        Input data matrix (sources × targets) or adjacency list DataFrame
    data_type : str
        'matrix' for 2D arrays, 'adjacency_list' for DataFrame with [source, target, value]
    figsize : tuple
        Figure dimensions in inches - LARGER for vivid display
    dpi : int
        Resolution in dots per inch - HIGHER for crisp rendering
    title : str
        Diagram title
    
    Returns
    -------
    matplotlib.figure.Figure
        Fully rendered vivid chord diagram
    """
    # Update global defaults with user parameters
    global CHORD_DIAGRAM_DEFAULTS
    CHORD_DIAGRAM_DEFAULTS.update({
        'figsize': figsize,
        'dpi': dpi,
        'big_gap': big_gap,
        'small_gap': small_gap,
        'track_height': 0.20,
        'grid_alpha': grid_alpha,
        'link_min_width': link_min_width,
        'link_max_width': link_max_width,
        'link_width_scale': link_width_scale,
        'link_alpha': link_alpha,
        'link_glow_intensity': link_glow_intensity,
        'arrow_length': arrow_length,
        'arrow_width': arrow_width,
        'diff_height': diff_height,
        'label_offset': sector_label_offset,
        'label_fontsize': sector_label_fontsize,
        'background_color': background_color,
        'grid_color': grid_color,
        'reduce_threshold': reduce_threshold,
        'sector_glow': sector_glow,
        'sector_glow_width': sector_glow_width,
        'sector_glow_alpha': sector_glow_alpha,
        'node_size_min': 150,
        'node_size_max': 800,
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
    
    # Limit number of links if too many
    if len(links) > max_links:
        links.sort(key=lambda x: x['value'], reverse=True)
        links = links[:max_links]
        print(f"Warning: Limited to top {max_links} links by value")
    
    # ========================================================================
    # DIAGRAM CONSTRUCTION - VIVID VERSION
    # ========================================================================
    diagram = VividChordDiagram(figsize=figsize, dpi=dpi, background_color=background_color)
    diagram.big_gap = big_gap
    diagram.small_gap = small_gap
    diagram.set_start_degree(start_degree)
    diagram.set_direction(direction == 'clockwise')
    diagram.background_color = background_color
    
    # Initialize sectors with grouping
    diagram.initialize_sectors(all_sectors, groups)
    
    # Generate sector colors based on palette choice
    if isinstance(sector_colors, str) and sector_colors == 'vivid':
        sector_color_list = VividColorPalettes.get_sector_colors(
            len(all_sectors), 
            palette=sector_color_palette
        )
        sector_colors = {sector: color for sector, color in zip(all_sectors, sector_color_list)}
    elif isinstance(sector_colors, dict):
        pass  # Use provided colors
    else:
        # Default to vivid palette
        sector_color_list = VividColorPalettes.get_sector_colors(len(all_sectors), palette='vivid')
        sector_colors = {sector: color for sector, color in zip(all_sectors, sector_color_list)}
    
    # Add tracks to all sectors with enhanced styling
    track_height = 0.20
    for sector in all_sectors:
        sector_color = sector_colors.get(sector, '#666666')
        
        for track_idx in range(tracks):
            track_color = track_colors[track_idx] if track_colors and track_idx < len(track_colors) else grid_color
            diagram.add_track(
                sector, 
                track_index=track_idx,
                height=track_heights[track_idx] if track_heights and track_idx < len(track_heights) else track_height,
                color=track_color,
                alpha=track_alpha,
                glow=sector_glow,
                border_color=sector_color if CHORD_DIAGRAM_DEFAULTS['show_node_borders'] else None
            )
    
    # ========================================================================
    # LINK PROCESSING & STYLING - VIVID VERSION
    # ========================================================================
    # Process link colors based on strategy
    link_color_map = {}
    link_cmap = None
    
    if isinstance(link_colors, str):
        if link_colors == 'group':
            for link in links:
                source_group = groups.get(link['source'], 0)
                link_color_map[(link['source'], link['target'])] = sector_colors.get(
                    link['source'] if source_group == 0 else link['target'],
                    '#888888'
                )
        elif link_colors == 'value' or link_colors == 'gradient':
            # Normalize values for colormap
            values = [link['value'] for link in links]
            vmin, vmax = min(values), max(values)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
            # Get colormap based on palette choice
            if link_color_palette == 'rainbow':
                link_cmap = VividColorPalettes.get_rainbow_palette()
            elif link_color_palette == 'fire':
                link_cmap = VividColorPalettes.get_fire_palette()
            elif link_color_palette == 'ocean':
                link_cmap = VividColorPalettes.get_ocean_palette()
            elif link_color_palette == 'electric':
                link_cmap = VividColorPalettes.get_electric_palette()
            elif link_color_palette == 'matrix':
                link_cmap = VividColorPalettes.get_matrix_palette()
            else:
                link_cmap = plt.cm.viridis
            
            for link in links:
                link_color_map[(link['source'], link['target'])] = link_cmap(norm(link['value']))
        elif link_colors == 'single_color':
            for link in links:
                link_color_map[(link['source'], link['target'])] = '#FF6B6B'  # Vivid coral
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
            link['zindex'] = 2 + (link['value'] / max_val) * 18  # Range: 2-20 for better layering
    elif link_zindex == 'random':
        for link in links:
            link['zindex'] = 2 + random.random() * 18
    elif isinstance(link_zindex, (list, np.ndarray)):
        for i, link in enumerate(links):
            link['zindex'] = link_zindex[i] if i < len(link_zindex) else 10.0
    else:
        for link in links:
            link['zindex'] = 10.0
    
    # Create links with enhanced vivid styling
    for link in links:
        source = link['source']
        target = link['target']
        value = link['value']
        
        # Check for highlighting
        is_highlighted = highlight_links and ((source, target) in highlight_links or 
                                             (target, source) in highlight_links)
        
        # Determine color
        color = link_color_map.get((source, target), '#AAAAAA')
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
        
        # Create the vivid link
        try:
            diagram.create_vivid_link(
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
                zindex=link['zindex'],
                add_glow=link_glow,
                add_border=link_border,
                gradient=link_gradient and link_colors == 'gradient'
            )
        except Exception as e:
            print(f"Warning: Could not draw link {source}→{target}: {str(e)}")
            continue
    
    # ========================================================================
    # FINALIZATION - ENHANCED
    # ========================================================================
    # Add sector labels with enhanced styling
    diagram.add_sector_labels(
        sector_labels,
        fontsize=sector_label_fontsize,
        offset=sector_label_offset,
        fontweight=sector_label_fontweight,
        shadow=sector_label_shadow,
        bg_alpha=0.95
    )
    
    # Add decorative sector nodes - ✅ NOW SAFE WITH FIXED IMPLEMENTATION
    if show_sector_nodes:
        diagram.add_sector_nodes(
            sector_colors,
            node_size_scale=sector_node_scale,
            show_labels=False
        )
    
    # Add legend if requested
    if show_legend and link_cmap is not None and links:
        values = [link['value'] for link in links]
        diagram.add_legend(
            title=legend_title,
            position=legend_position,
            cmap=link_cmap,
            vmin=min(values),
            vmax=max(values),
            fontsize=CHORD_DIAGRAM_DEFAULTS['legend_fontsize']
        )
    
    # Calculate and add statistics
    if show_statistics:
        stats = {
            'Total Links': len(links),
            'Total Value': sum(link['value'] for link in links),
            'Max Link': max(link['value'] for link in links) if links else 0,
            'Avg Link': np.mean([link['value'] for link in links]) if links else 0,
            'Sectors': len(all_sectors)
        }
        diagram.add_statistics_box(
            stats,
            position=statistics_position,
            fontsize=statistics_fontsize
        )
    
    # Finalize and return figure
    fig = diagram.finalize(
        title=title,
        show_frame=show_frame,
        background_color=background_color,
        show_grid=show_grid
    )
    
    return fig


# ============================================================================
# ENHANCED STREAMLIT APPLICATION - VIVID NETWORK VISUALIZATION
# ============================================================================

def create_vivid_streamlit_app():
    """Production-ready Streamlit application for vivid chord diagram creation."""
    
    # Page configuration - OPTIMIZED FOR LARGE DISPLAYS
    st.set_page_config(
        page_title="🎨 Vivid Network Chord Diagrams",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced UI - OPTIMIZED FOR VIVID DISPLAY
    st.markdown("""
    <style>
    /* Global styles */
    .main {
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #0A0F1A 0%, #1A2332 100%);
    }
    .stApp {
        background-color: #0A0F1A;
    }
    
    /* Header styling */
    h1 {
        color: #FFFFFF;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(255, 255, 255, 0.3);
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        color: #A0AEC0;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        text-shadow: 0 1px 5px rgba(0, 0, 0, 0.5);
    }
    
    /* Section dividers */
    .section-divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #4A5568, transparent);
        margin: 2rem 0;
    }
    
    /* Cards and containers */
    .stCard {
        background: linear-gradient(135deg, #1A2332 0%, #2A3B5C 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
        border: 1px solid #4A5568;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #EE5A24);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
        background: linear-gradient(90deg, #FF5252, #E54A1A);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #2D3748, #4A5568) !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.2rem !important;
        color: #E2E8F0 !important;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        background: rgba(26, 35, 50, 0.7);
        padding: 1rem;
        border-radius: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: nowrap;
        border-radius: 12px;
        background: rgba(45, 55, 72, 0.5);
        color: #A0AEC0;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4ECDC4, #45B7D1);
        color: white;
        box-shadow: 0 4px 12px rgba(78, 205, 196, 0.3);
    }
    
    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, #1A2332, #2A3B5C);
        padding: 1.2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #4A5568;
    }
    .stMetric > div {
        color: #FFFFFF !important;
    }
    
    /* Selectbox and sliders */
    .stSelectbox > div > div,
    .stSlider > div > div {
        color: #E2E8F0 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.95rem;
        margin-top: 4rem;
        padding: 2rem;
        border-top: 1px solid #2D3748;
        background: rgba(26, 35, 50, 0.5);
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #FF6B6B #FF6B6B #FF6B6B transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🎨 Vivid Network Chord Diagrams")
    st.markdown(
        '<p class="subtitle">Ultra-high resolution circular network visualizations with '
        'vivid colors, glow effects, and enhanced readability. Perfect for presentations, '
        'publications, and large displays.</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60/FF6B6B/FFFFFF?text=VIVID+NETWORKS", 
                use_column_width=True)
        st.markdown("### 📊 Data Configuration")
        
        # Data source selection - ✅ LOADING CSV WITH OS.PATH.JOIN
        use_example = st.checkbox("✨ Use dendrites attributes dataset", value=True, key="use_example")
        
        if use_example:
            try:
                # ✅ SAFE FILE PATH CONSTRUCTION USING OS.PATH.JOIN
                current_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(current_dir, "dendrites_attributes.csv")
                
                # Fallback to working directory if needed
                if not os.path.exists(file_path):
                    file_path = os.path.join(os.getcwd(), "dendrites_attributes.csv")
                
                # Load and process CSV
                raw_data = pd.read_csv(file_path)
                st.success(f"✅ Loaded dendrites_attributes.csv ({len(raw_data)} rows)")
                
                # Compute correlation matrix for chord diagram
                data_for_corr = raw_data.drop(columns=['step'], errors='ignore')
                corr_matrix = data_for_corr.corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)  # Zero diagonal
                
                data = corr_matrix
                data_type = "matrix"
                st.info(f"🔄 Converted to correlation matrix ({data.shape[0]}×{data.shape[1]})")
                
            except Exception as e:
                st.error(f"❌ Error loading dendrites_attributes.csv: {str(e)}")
                st.warning("⚠️ Falling back to generated example data")
                
                # Fallback to generated data
                np.random.seed(42)
                n = 15
                matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            matrix[i, j] = 1.0
                        elif abs(i - j) <= 3:
                            matrix[i, j] = 0.9 - abs(i - j) * 0.2
                        else:
                            matrix[i, j] = np.random.uniform(0.1, 0.4)
                
                for i in range(n):
                    for j in range(i+1, n):
                        if np.random.random() > 0.6:
                            matrix[i, j] *= 1.8
                
                row_names = [f"Gene_{i+1:02d}" for i in range(n)]
                col_names = [f"Pathway_{j+1:02d}" for j in range(n)]
                data = pd.DataFrame(matrix, index=row_names, columns=col_names)
                data_type = "matrix"
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
                st.info("👆 Upload your data or use the dendrites dataset")
                st.stop()
        
        # Quick presets for vivid styles
        st.markdown("---")
        st.markdown("### ⚡ Vivid Presets")
        preset = st.selectbox("Apply vivid style preset", 
                            ["Default", "Fire Network", "Ocean Flow", "Rainbow Web", 
                             "Electric Grid", "Matrix Code", "Royal Network"])
        if preset == "Fire Network":
            st.session_state.update({
                'sector_color_palette': 'fire',
                'link_color_palette': 'fire',
                'background_color': '#000000',
                'directional': True,
                'link_glow': True
            })
        elif preset == "Ocean Flow":
            st.session_state.update({
                'sector_color_palette': 'ocean',
                'link_color_palette': 'ocean',
                'background_color': '#001122',
                'directional': True,
                'sector_glow': True
            })
        elif preset == "Rainbow Web":
            st.session_state.update({
                'sector_color_palette': 'rainbow',
                'link_color_palette': 'rainbow',
                'background_color': '#111111',
                'link_gradient': True,
                'link_glow': True
            })
        elif preset == "Electric Grid":
            st.session_state.update({
                'sector_color_palette': 'electric',
                'link_color_palette': 'electric',
                'background_color': '#000000',
                'link_border': True,
                'sector_glow': True
            })
        elif preset == "Matrix Code":
            st.session_state.update({
                'sector_color_palette': 'matrix',
                'link_color_palette': 'matrix',
                'background_color': '#000000',
                'sector_label_fontcolor': '#00FF00',
                'link_glow': True
            })
        elif preset == "Royal Network":
            st.session_state.update({
                'sector_color_palette': 'vivid',
                'link_color_palette': 'rainbow',
                'background_color': '#0A001A',
                'link_border': True,
                'link_border_color': '#FFD700'
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
        st.header("Ultra-High Resolution Network Visualization")
        
        # Parameter configuration in expanders
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### ⚙️ Rendering Controls")
            
            with st.expander("🧭 Layout & Orientation", expanded=True):
                start_degree = st.slider("Starting angle (°)", 0, 359, 0, 15)
                direction = st.selectbox("Direction", ["clockwise", "counter-clockwise"])
                big_gap = st.slider("Group gap (°)", 0, 45, 15, 1)
                small_gap = st.slider("Sector gap (°)", 0, 15, 3, 1)
            
            with st.expander("🎨 Color & Styling", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    sector_color_palette = st.selectbox(
                        "Sector palette",
                        ["vivid", "fire", "ocean", "rainbow", "electric", "matrix"],
                        index=["vivid", "fire", "ocean", "rainbow", "electric", "matrix"].index(
                            st.session_state.get('sector_color_palette', 'vivid')
                        )
                    )
                    link_color_palette = st.selectbox(
                        "Link palette",
                        ["rainbow", "fire", "ocean", "electric", "matrix", "group", "single"],
                        index=["rainbow", "fire", "ocean", "electric", "matrix", "group", "single"].index(
                            st.session_state.get('link_color_palette', 'rainbow')
                        )
                    )
                with col_b:
                    bg_color = st.color_picker("Background", 
                                              st.session_state.get('background_color', '#0A0F1A'))
                    link_alpha = st.slider("Link opacity", 0.3, 1.0, 0.85, 0.05)
                
                show_sector_nodes = st.checkbox("Show sector nodes", value=True)
                sector_node_scale = st.slider("Node size scale", 0.5, 2.0, 1.2, 0.1)
            
            with st.expander("✨ Visual Effects", expanded=True):
                link_glow = st.checkbox("Link glow effect", 
                                       value=st.session_state.get('link_glow', True))
                link_glow_intensity = st.slider("Glow intensity", 0.5, 5.0, 2.5, 0.5)
                
                link_border = st.checkbox("Link borders", 
                                         value=st.session_state.get('link_border', True))
                link_border_width = st.slider("Border width", 0.2, 2.0, 0.8, 0.2)
                
                sector_glow = st.checkbox("Sector glow", 
                                         value=st.session_state.get('sector_glow', True))
                sector_glow_width = st.slider("Sector glow width", 2, 20, 8, 2)
            
            with st.expander("➡️ Directional Features", expanded=False):
                directional = st.checkbox("Enable directional flows", 
                                         value=st.session_state.get('directional', False))
                
                direction_type = ["diffHeight", "arrows"]  # Default value
                arrow_size = 1.5  # Default value
                
                if directional:
                    direction_type = st.multiselect(
                        "Direction indicators",
                        ["diffHeight", "arrows"],
                        default=["diffHeight", "arrows"]
                    )
                    arrow_size = st.slider("Arrow size", 0.5, 3.0, 1.5, 0.1)
            
            with st.expander("⚡ Performance", expanded=False):
                reduce_threshold = st.slider("Min link value", 0.0, 0.5, 0.01, 0.01)
                max_links = st.slider("Max links to display", 100, 2000, 1000, 100)
                st.caption("Links below threshold will be hidden")
        
        with col1:
            # Generate diagram with error handling
            try:
                with st.spinner("🎨 Rendering vivid network diagram..."):
                    params = {
                        'data': data,
                        'data_type': data_type,
                        'figsize': (24, 24),
                        'dpi': 300,
                        'title': f"Dendrites Attributes Correlation • {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        'start_degree': start_degree,
                        'direction': direction,
                        'big_gap': big_gap,
                        'small_gap': small_gap,
                        'sector_color_palette': sector_color_palette,
                        'link_color_palette': link_color_palette,
                        'link_alpha': link_alpha,
                        'link_glow': link_glow,
                        'link_glow_intensity': link_glow_intensity,
                        'link_border': link_border,
                        'link_border_width': link_border_width,
                        'background_color': bg_color,
                        'sector_glow': sector_glow,
                        'sector_glow_width': sector_glow_width,
                        'show_sector_nodes': show_sector_nodes,
                        'sector_node_scale': sector_node_scale,
                        'reduce_threshold': reduce_threshold,
                        'max_links': max_links,
                        'directional': directional,
                        'direction_type': direction_type,
                        'arrow_length': 0.12 * arrow_size,
                        'arrow_width': 0.07 * arrow_size,
                        'symmetric': False,
                        'link_gradient': (link_color_palette == 'rainbow'),
                        'show_legend': True,
                        'show_statistics': True
                    }
                    
                    fig = create_vivid_chord_diagram(**params)
                    st.pyplot(fig, use_container_width=True, clear_figure=True)
                    st.session_state['current_figure'] = fig
                    
            except Exception as e:
                st.error(f"❌ Rendering error: {str(e)}")
                with st.expander("🔍 Show traceback"):
                    st.code(traceback.format_exc())
        
        # Interactive legend
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📌 Visualization Features")
        col_leg1, col_leg2, col_leg3, col_leg4 = st.columns(4)
        with col_leg1:
            st.markdown("✨ **Glow Effects**: Links pulse with energy")
        with col_leg2:
            st.markdown("🎨 **Vivid Colors**: High-contrast palettes")
        with col_leg3:
            st.markdown("🎯 **Node Sizing**: Proportional to connectivity")
        with col_leg4:
            if directional:
                st.markdown("➡️ **Directional**: Arrows show flow")
            else:
                st.markdown("🔄 **Undirected**: Symmetric relationships")
    
    # ============================================================================
    # DATA EXPLORER TAB
    # ============================================================================
    with tab_data:
        st.header("Dataset Analysis")
        
        if data_type == "matrix":
            col_stats1, col_stats2, col_stats3, col_stats4, col_stats5 = st.columns(5)
            with col_stats1:
                st.metric("Rows (Sources)", data.shape[0])
            with col_stats2:
                st.metric("Columns (Targets)", data.shape[1])
            with col_stats3:
                st.metric("Total Links", f"{np.sum(np.abs(data.values) > reduce_threshold):,}")
            with col_stats4:
                density = np.mean(np.abs(data.values) > reduce_threshold) * 100
                st.metric("Density", f"{density:.1f}%")
            with col_stats5:
                st.metric("Max Value", f"{data.values.max():.3f}")
            
            st.markdown("#### 🔍 Data Preview")
            # Style with gradient for better visibility
            styled_data = data.style.background_gradient(
                cmap='RdBu_r', 
                axis=None, 
                vmin=-1, 
                vmax=1
            ).format("{:.3f}")
            st.dataframe(styled_data, use_container_width=True)
            
            st.markdown("#### 📈 Value Distribution")
            flat_vals = data.values.flatten()
            fig_hist, ax_hist = plt.subplots(figsize=(10, 4), facecolor='#0A0F1A')
            ax_hist.set_facecolor('#0A0F1A')
            n, bins, patches = ax_hist.hist(flat_vals, bins=60, 
                                           color='#FF6B6B', alpha=0.8, 
                                           edgecolor='white', linewidth=0.5)
            ax_hist.set_xlabel('Value', color='white', fontsize=12)
            ax_hist.set_ylabel('Frequency', color='white', fontsize=12)
            ax_hist.set_title('Distribution of Link Values', color='white', fontsize=14, fontweight='bold')
            ax_hist.grid(True, alpha=0.2, color='#4A5568')
            ax_hist.tick_params(colors='white')
            st.pyplot(fig_hist, use_container_width=True)
        else:
            st.markdown(f"**Links**: {len(data)}")
            st.markdown(f"**Unique Sources**: {data.iloc[:,0].nunique()}")
            st.markdown(f"**Unique Targets**: {data.iloc[:,1].nunique()}")
            st.dataframe(data.head(20), use_container_width=True)
    
    # ============================================================================
    # USER GUIDE TAB
    # ============================================================================
    with tab_guide:
        st.header("📘 Complete User Guide")
        
        st.markdown("""
        ### 🌟 Vivid Network Features
        
        #### Ultra-High Resolution Display
        - **24×24 inch canvas** at **300 DPI** for publication-quality output
        - Crisp rendering suitable for large-format printing and presentations
        - Optimized for 4K displays and high-resolution projectors
        
        #### Vivid Visual Effects
        
        ##### Glow Effects
        - **Link glow**: Multi-layer glow around connections for depth
        - **Sector glow**: Radiant halos around sector nodes
        - **Highlight glow**: Intense pulsing for emphasized links
        
        ##### Enhanced Colors
        - **Fire palette**: Black → Red → Orange → Yellow → White gradient
        - **Ocean palette**: Deep blue → Cyan → White gradient
        - **Rainbow palette**: Full spectrum for maximum differentiation
        - **Electric palette**: Neon colors on dark background
        - **Matrix palette**: Green code-inspired aesthetic
        
        ##### Node Visualization
        - **Dynamic sizing**: Node size proportional to connectivity degree
        - **Decorative borders**: White outlines for contrast
        - **Position markers**: Visual anchors at sector locations
        
        #### Directional Flow Visualization
        - **Height Differentiation**: Source links attach lower, targets higher
        - **Multiple arrows**: Strong links get multiple directional indicators
        - **Enhanced arrows**: Larger, more visible arrowheads
        
        #### Sector Organization
        - **Group Separation**: Automatic visual grouping with larger gaps
        - **Custom Ordering**: Control sector sequence for optimal pattern visibility
        - **Multi-Track Architecture**: Layer additional data dimensions
        
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
        
        ### 💡 Pro Tips for Vivid Networks
        
        1. **For presentations**: Use Fire or Electric palettes on black background
        2. **For publications**: Use Rainbow or Ocean palettes with white borders
        3. **For flow diagrams**: Enable directional links with both height differentiation and arrows
        4. **For correlation matrices**: Use symmetric mode with value-based coloring
        5. **Reduce visual clutter**: Adjust the "Min link value" threshold to hide weak connections
        6. **Highlight key relationships**: Use the API's `highlight_links` parameter
        7. **Maximize impact**: Enable glow effects and node sizing for dramatic visuals
        
        ### 🔬 Use Cases
        
        - **Bioinformatics**: Gene-pathway interactions, protein-protein networks
        - **Flow Analysis**: Migration patterns, financial transactions, user journey mapping
        - **Correlation Analysis**: Feature relationships in ML datasets
        - **Social Networks**: Community detection and relationship mapping
        - **Transportation**: Traffic flow, airline routes, shipping lanes
        - **Energy Grids**: Power distribution networks
        
        ### 🚀 Performance Notes
        
        - Handles datasets up to **1000+ links** smoothly
        - For larger networks (>2000 links):
            * Increase the reduction threshold
            * Pre-filter weak connections
            * Consider sampling or aggregation
        - Rendering time: 5-15 seconds for 500 links at 300 DPI
        - Memory usage: ~500MB for 1000-link diagram
        
        ### 📊 Export Recommendations
        
        - **PNG (300 DPI)**: Best for presentations and web (raster)
        - **SVG**: Ideal for editing in Illustrator/Inkscape (vector)
        - **PDF**: Required for academic publications (vector quality)
        - **TIFF (600 DPI)**: For high-end printing and posters
        """)
    
    # ============================================================================
    # EXPORT TAB - ENHANCED
    # ============================================================================
    with tab_export:
        st.header("📤 Export Options")
        
        if 'current_figure' not in st.session_state:
            st.info("💡 Create a vivid diagram first, then export it here")
        else:
            fig = st.session_state['current_figure']
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                st.markdown("#### 🖼️ High-Resolution Image Formats")
                dpi = st.slider("Resolution (DPI)", 150, 600, 300, 50)
                st.info(f"Current canvas: 24×24 inches @ {dpi} DPI = {24*dpi}×{24*dpi} pixels")
                
                # PNG Export
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format='png', dpi=dpi, 
                          bbox_inches='tight', facecolor=bg_color, 
                          transparent=False, pad_inches=0.5)
                buf_png.seek(0)
                st.download_button(
                    label="⬇️ Download PNG (Ultra HD)",
                    data=buf_png,
                    file_name=f"dendrites_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    key="png_download",
                    help="Best for presentations and web use"
                )
                
                # TIFF Export (for print)
                buf_tiff = io.BytesIO()
                fig.savefig(buf_tiff, format='tiff', dpi=max(dpi, 300),
                          bbox_inches='tight', facecolor=bg_color,
                          pad_inches=0.5)
                buf_tiff.seek(0)
                st.download_button(
                    label="⬇️ Download TIFF (Print Ready)",
                    data=buf_tiff,
                    file_name=f"dendrites_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff",
                    mime="image/tiff",
                    key="tiff_download",
                    help="Best for high-quality printing and posters"
                )
            
            with col_exp2:
                st.markdown("#### 📄 Vector & Publication Formats")
                
                # PDF Export
                buf_pdf = io.BytesIO()
                fig.savefig(buf_pdf, format='pdf', bbox_inches='tight',
                          facecolor=bg_color, pad_inches=0.5)
                buf_pdf.seek(0)
                st.download_button(
                    label="⬇️ Download PDF (Publication Quality)",
                    data=buf_pdf,
                    file_name=f"dendrites_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="pdf_download",
                    help="Required for academic publications"
                )
                
                # SVG Export (vector)
                buf_svg = io.BytesIO()
                fig.savefig(buf_svg, format='svg', bbox_inches='tight',
                          facecolor=bg_color, pad_inches=0.5)
                buf_svg.seek(0)
                st.download_button(
                    label="⬇️ Download SVG (Editable Vector)",
                    data=buf_svg,
                    file_name=f"dendrites_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                    mime="image/svg+xml",
                    key="svg_download",
                    help="Best for editing in Illustrator or Inkscape"
                )
            
            st.markdown("#### 💡 Export Tips")
            st.info(
                """
                **For Presentations:**
                - Use PNG at 150-300 DPI
                - Black background works best on projectors
                
                **For Publications:**
                - Use PDF or SVG (vector formats)
                - Ensure color scheme is printer-friendly
                - Consider grayscale version for print
                
                **For Web:**
                - Use PNG at 150 DPI
                - Optimize file size with external tools
                
                **For Large Format Printing:**
                - Use TIFF at 300-600 DPI
                - Verify colors in CMYK if needed for print
                """
            )
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Vivid Network Chord Diagrams • 24×24 inch • 300 DPI • Ultra-High Resolution</p>
        <p>💡 Pro Tip: For maximum visual impact, use glow effects with Fire or Electric palettes 
        on dark backgrounds. Enable node sizing to show hub sectors!</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Application entry point."""
    try:
        create_vivid_streamlit_app()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
