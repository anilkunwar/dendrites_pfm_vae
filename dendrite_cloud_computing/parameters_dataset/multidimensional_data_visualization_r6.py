# -*- coding: utf-8 -*-
"""
Enhanced Vivid Chord Diagram Visualization with Light Theme
Ultra-high resolution circular network visualizations with professional light theme
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
import hashlib
import time
import json
import pickle
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
import base64

warnings.filterwarnings('ignore')

# ============================================================================
# ENHANCED CONFIGURATION & CONSTANTS - LIGHT THEME
# ============================================================================

class ColorTheme(Enum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"

@dataclass
class ThemeConfig:
    """Configuration for different color themes"""
    name: str
    background_color: str
    grid_color: str
    text_color: str
    panel_bg: str
    panel_border: str
    highlight_color: str
    node_border_color: str
    link_border_color: str
    label_bg_color: str
    
    @classmethod
    def get_theme(cls, theme_name: str):
        """Get theme configuration by name"""
        themes = {
            "light": ThemeConfig(
                name="light",
                background_color='#FFFFFF',
                grid_color='#E0E0E0',
                text_color='#333333',
                panel_bg='#F8F9FA',
                panel_border='#D1D5DB',
                highlight_color='#2563EB',
                node_border_color='#4B5563',
                link_border_color='#374151',
                label_bg_color='rgba(255, 255, 255, 0.9)'
            ),
            "dark": ThemeConfig(
                name="dark",
                background_color='#0F172A',
                grid_color='#1E293B',
                text_color='#F1F5F9',
                panel_bg='#1E293B',
                panel_border='#334155',
                highlight_color='#60A5FA',
                node_border_color='#CBD5E1',
                link_border_color='#E2E8F0',
                label_bg_color='rgba(15, 23, 42, 0.9)'
            ),
            "professional": ThemeConfig(
                name="professional",
                background_color='#F8FAFC',
                grid_color='#E2E8F0',
                text_color='#1E293B',
                panel_bg='#FFFFFF',
                panel_border='#CBD5E1',
                highlight_color='#3B82F6',
                node_border_color='#475569',
                link_border_color='#334155',
                label_bg_color='rgba(255, 255, 255, 0.95)'
            ),
            "pastel": ThemeConfig(
                name="pastel",
                background_color='#FEFCE8',
                grid_color='#FDE68A',
                text_color='#78350F',
                panel_bg='#FEF3C7',
                panel_border='#F59E0B',
                highlight_color='#8B5CF6',
                node_border_color='#7C3AED',
                link_border_color='#6D28D9',
                label_bg_color='rgba(254, 252, 232, 0.95)'
            ),
            "ocean": ThemeConfig(
                name="ocean",
                background_color='#F0F9FF',
                grid_color='#BAE6FD',
                text_color='#0C4A6E',
                panel_bg='#E0F2FE',
                panel_border='#38BDF8',
                highlight_color='#0369A1',
                node_border_color='#0EA5E9',
                link_border_color='#0284C7',
                label_bg_color='rgba(240, 249, 255, 0.95)'
            )
        }
        return themes.get(theme_name, themes["light"])

# LIGHT THEME DEFAULTS
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
    'link_border_color': '#374151', # Light theme: dark gray border
    
    # ENHANCED DIRECTIONAL FEATURES
    'arrow_length': 0.18,          # Increased from 0.12
    'arrow_width': 0.10,           # Increased from 0.06
    'diff_height': 0.06,           # Increased from 0.03
    
    # IMPROVED LABELING
    'label_offset': 0.30,          # Increased from 0.18
    'label_fontsize': 18,          # Increased from 13
    'label_fontweight': 'bold',
    'label_shadow': False,         # Changed to False for light theme
    'label_bg_alpha': 0.95,
    
    # LIGHT THEME COLORS
    'background_color': '#FFFFFF', # White background for light theme
    'grid_color': '#E0E0E0',       # Light gray grid
    'highlight_alpha': 0.98,
    'highlight_glow': 2.0,         # Reduced for light theme
    
    # NETWORK ENHANCEMENTS
    'reduce_threshold': 0.001,     # Lower threshold to show more links
    'node_size_min': 150,          # NEW: Minimum node size
    'node_size_max': 800,          # NEW: Maximum node size
    'show_node_borders': True,
    'node_border_width': 2,        # Slightly thinner for light theme
    'sector_glow': False,          # Disabled for light theme
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
    'simplify_curves': False,      # Keep detailed curves
    
    # LIGHT THEME SPECIFIC
    'text_color': '#333333',       # Dark text for light background
    'panel_bg': '#F8F9FA',         # Light panel background
    'panel_border': '#D1D5DB',     # Panel border color
    'highlight_color': '#2563EB',  # Professional blue highlight
    'node_border_color': '#4B5563' # Dark gray node borders
}

# ============================================================================
# CACHE MANAGEMENT SYSTEM
# ============================================================================

class CacheManager:
    """Enhanced cache management system for performance optimization"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_data(file_path: str, is_example: bool = True):
        """Cache data loading operations"""
        try:
            if is_example:
                # For example data, generate once and cache
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
                return data
            else:
                # For uploaded files, cache by file content hash
                return pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=600, show_spinner=False)
    def compute_correlation(data: pd.DataFrame):
        """Cache correlation matrix computation"""
        try:
            data_for_corr = data.drop(columns=['step'], errors='ignore')
            corr_matrix = data_for_corr.corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            return corr_matrix
        except Exception as e:
            st.error(f"Error computing correlation: {str(e)}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def generate_figure_hash(params: dict):
        """Generate a hash for figure parameters to detect changes"""
        # Create a stable string representation of parameters
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    @staticmethod
    def get_cached_figure(figure_hash: str):
        """Get cached figure if available"""
        if 'figure_cache' not in st.session_state:
            st.session_state.figure_cache = {}
        
        return st.session_state.figure_cache.get(figure_hash)
    
    @staticmethod
    def cache_figure(figure_hash: str, figure):
        """Cache a figure"""
        if 'figure_cache' not in st.session_state:
            st.session_state.figure_cache = {}
        
        # Limit cache size
        if len(st.session_state.figure_cache) > 10:
            # Remove oldest entry
            oldest_key = next(iter(st.session_state.figure_cache))
            del st.session_state.figure_cache[oldest_key]
        
        st.session_state.figure_cache[figure_hash] = figure

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

class SessionStateManager:
    """Manage session state to prevent unnecessary re-runs"""
    
    @staticmethod
    def initialize_state():
        """Initialize all session state variables"""
        defaults = {
            # Data state
            'current_data': None,
            'data_type': 'matrix',
            'data_hash': None,
            
            # Visualization parameters
            'viz_params': {},
            'current_figure': None,
            'figure_hash': None,
            
            # UI state
            'theme': 'light',
            'preset': 'Default',
            'sidebar_collapsed': False,
            
            # Performance metrics
            'render_count': 0,
            'last_render_time': None,
            'avg_render_time': 0,
            
            # User preferences
            'auto_render': True,
            'high_quality': True,
            'show_help': True,
            
            # Export settings
            'export_format': 'png',
            'export_dpi': 300,
            
            # Analytics
            'page_views': 0,
            'interaction_count': 0,
            'last_interaction': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Increment page views
        st.session_state.page_views += 1
    
    @staticmethod
    def update_interaction():
        """Update interaction tracking"""
        st.session_state.interaction_count += 1
        st.session_state.last_interaction = datetime.now()
    
    @staticmethod
    def should_render(new_params: dict) -> bool:
        """Determine if we should re-render based on parameter changes"""
        old_params = st.session_state.get('viz_params', {})
        
        # If no previous params, render
        if not old_params:
            return True
        
        # Check if critical parameters changed
        critical_params = [
            'data_hash', 'sector_color_palette', 'link_color_palette',
            'background_color', 'directional', 'link_glow'
        ]
        
        for param in critical_params:
            if old_params.get(param) != new_params.get(param):
                return True
        
        # For other parameters, use hash comparison
        old_hash = CacheManager.generate_figure_hash(old_params)
        new_hash = CacheManager.generate_figure_hash(new_params)
        
        return old_hash != new_hash
    
    @staticmethod
    def get_performance_stats():
        """Get rendering performance statistics"""
        return {
            'render_count': st.session_state.render_count,
            'avg_render_time': st.session_state.avg_render_time,
            'page_views': st.session_state.page_views,
            'interaction_count': st.session_state.interaction_count
        }

# ============================================================================
# LIGHT THEME COLOR PALETTES
# ============================================================================

class LightThemeColorPalettes:
    """Collection of light theme color palettes for network visualization"""
    
    @staticmethod
    def create_gradient_cmap(colors, name='custom_gradient'):
        """Create smooth gradient colormap from color list"""
        return LinearSegmentedColormap.from_list(name, colors, N=256)
    
    @staticmethod
    def get_light_palette(n_colors=20):
        """Get light-friendly distinguishable colors"""
        # Use Set3 or Set2 for light backgrounds
        return sns.color_palette("Set3", n_colors)
    
    @staticmethod
    def get_pastel_palette():
        """Soft pastel colors"""
        return LightThemeColorPalettes.create_gradient_cmap([
            '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', 
            '#BAE1FF', '#D0BAFF', '#FFBAF1', '#FFFFFF'
        ], 'pastel')
    
    @staticmethod
    def get_professional_palette():
        """Professional color palette for business/analytics"""
        return LightThemeColorPalettes.create_gradient_cmap([
            '#1E3A8A', '#2563EB', '#3B82F6', '#60A5FA',
            '#93C5FD', '#BFDBFE', '#DBEAFE', '#FFFFFF'
        ], 'professional')
    
    @staticmethod
    def get_warm_palette():
        """Warm colors suitable for light backgrounds"""
        return LightThemeColorPalettes.create_gradient_cmap([
            '#7C2D12', '#DC2626', '#EA580C', '#F59E0B',
            '#FBBF24', '#FDE68A', '#FEF3C7', '#FFFFFF'
        ], 'warm')
    
    @staticmethod
    def get_cool_palette():
        """Cool colors suitable for light backgrounds"""
        return LightThemeColorPalettes.create_gradient_cmap([
            '#1E3A8A', '#1D4ED8', '#3B82F6', '#60A5FA',
            '#93C5FD', '#BFDBFE', '#DBEAFE', '#FFFFFF'
        ], 'cool')
    
    @staticmethod
    def get_earth_palette():
        """Earth tones suitable for light backgrounds"""
        return LightThemeColorPalettes.create_gradient_cmap([
            '#422006', '#854D0E', '#A16207', '#CA8A04',
            '#EAB308', '#FDE047', '#FEF08A', '#FEFCE8'
        ], 'earth')
    
    @staticmethod
    def get_sector_colors(n_sectors, palette='light'):
        """Generate sector colors based on palette choice for light theme"""
        if palette == 'light':
            colors = LightThemeColorPalettes.get_light_palette(n_sectors)
        elif palette == 'pastel':
            cmap = LightThemeColorPalettes.get_pastel_palette()
            colors = [cmap(i / n_sectors) for i in range(n_sectors)]
        elif palette == 'professional':
            cmap = LightThemeColorPalettes.get_professional_palette()
            colors = [cmap(i / n_sectors) for i in range(n_sectors)]
        elif palette == 'warm':
            cmap = LightThemeColorPalettes.get_warm_palette()
            colors = [cmap(i / n_sectors) for i in range(n_sectors)]
        elif palette == 'cool':
            cmap = LightThemeColorPalettes.get_cool_palette()
            colors = [cmap(i / n_sectors) for i in range(n_sectors)]
        elif palette == 'earth':
            cmap = LightThemeColorPalettes.get_earth_palette()
            colors = [cmap(i / n_sectors) for i in range(n_sectors)]
        else:  # Default Set3
            cmap = plt.cm.Set3
            colors = [cmap(i % 12) for i in range(n_sectors)]
        
        return colors

# ============================================================================
# ENHANCED CHORD DIAGRAM ENGINE - LIGHT THEME OPTIMIZED
# ============================================================================

class LightThemeChordDiagram:
    """
    Enhanced chord diagram renderer optimized for light themes.
    
    Features:
        - Ultra-large canvas (24x24 inches) at 300 DPI
        - Light theme color schemes with high contrast
        - Optimized visual effects for light backgrounds
        - Enhanced directional indicators
        - Node sizing based on connectivity
        - Radial gradients and visual effects
        - Comprehensive legends and statistics
    """
    
    def __init__(self, figsize: Tuple[int, int] = (24, 24), dpi: int = 300,
                 background_color: str = '#FFFFFF', theme_config: ThemeConfig = None):
        """
        Initialize enhanced chord diagram canvas.
        
        Parameters
        ----------
        figsize : tuple
            Figure dimensions (width, height) in inches - LARGER for vivid display
        dpi : int
            Resolution in dots per inch - HIGHER for crisp rendering
        background_color : str
            Canvas background color - LIGHT for professional look
        theme_config : ThemeConfig
            Theme configuration object
        """
        self.figsize = figsize
        self.dpi = dpi
        self.background_color = background_color
        self.theme_config = theme_config or ThemeConfig.get_theme("light")
        
        # Create figure with larger dimensions
        self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=background_color)
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_theta_zero_location("N")  # 0° at top
        self.ax.set_theta_direction(-1)       # Clockwise by default
        
        # Data structures
        self.sectors: List[str] = []
        self.sector_dict: Dict[str, Dict] = {}
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
                         color: Union[str, Tuple] = '#E0E0E0',  # Lighter for light theme
                         alpha: float = 0.15,
                         track_height: float = 0.20, 
                         track_index: int = 0,
                         glow: bool = False,  # Disabled for light theme
                         border_color: Optional[str] = None,
                         border_width: float = 2.0) -> Dict[str, float]:
        """
        Render a single sector track with light-theme optimized visual effects.
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
        
        # Add subtle border for light theme
        if border_color and CHORD_DIAGRAM_DEFAULTS['show_node_borders']:
            border_alpha = min(1.0, alpha * 1.2)
            border_poly = plt.Polygon(vertices, facecolor='none', 
                                     edgecolor=border_color, alpha=border_alpha,
                                     linewidth=border_width, zorder=1.5)
            self.ax.add_patch(border_poly)
        
        return {'inner': r_inner, 'outer': r_outer, 'mid_angle': np.radians(angles['mid'])}
    
    def create_light_link(self, source: str, target: str, value: float,
                         source_track: int = 0, target_track: int = 0,
                         color: Union[str, Tuple] = '#3B82F6',  # Professional blue
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
        Create a light-theme optimized curved link between two sectors.
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
        t = np.linspace(0, 1, 100)
        theta_curve = (1 - t)**2 * source_rad + 2 * (1 - t) * t * control_angle + t**2 * target_rad
        r_curve = (1 - t)**2 * source_r + 2 * (1 - t) * t * control_r + t**2 * target_r
        
        # Normalize angles to [0, 2π)
        theta_curve = theta_curve % (2 * np.pi)
        
        # Calculate base linewidth
        base_width = max(CHORD_DIAGRAM_DEFAULTS['link_min_width'], 
                        min(CHORD_DIAGRAM_DEFAULTS['link_max_width'], 
                            value * CHORD_DIAGRAM_DEFAULTS['link_width_scale']))
        
        # Add subtle glow effect for light theme
        if add_glow and CHORD_DIAGRAM_DEFAULTS['link_glow_intensity'] > 0:
            glow_width = base_width * CHORD_DIAGRAM_DEFAULTS['link_glow_intensity'] * 0.5  # Reduced for light
            glow_alpha = CHORD_DIAGRAM_DEFAULTS['link_glow_alpha'] * 0.5  # Reduced for light
            
            # Subtle single-layer glow
            glow_line, = self.ax.plot(theta_curve, r_curve,
                                    color=color, 
                                    alpha=glow_alpha * (alpha if not highlight else 1.0),
                                    linewidth=glow_width * 1.5,
                                    solid_capstyle='round',
                                    solid_joinstyle='round',
                                    zorder=zindex - 0.5)
        
        # Add border for contrast
        if add_border and CHORD_DIAGRAM_DEFAULTS['link_border_width'] > 0:
            border_line, = self.ax.plot(theta_curve, r_curve,
                                      color=CHORD_DIAGRAM_DEFAULTS['link_border_color'],
                                      alpha=min(0.8, alpha * 1.1),  # Reduced alpha for light
                                      linewidth=base_width + CHORD_DIAGRAM_DEFAULTS['link_border_width'],
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
            arrow_positions = [0.4, 0.6, 0.8] if value > 0.5 else [0.6]
            
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
                        alpha=min(1.0, alpha * 1.3),
                        linewidth=base_width * 0.6,
                        mutation_scale=25,
                        zorder=zindex + 0.5
                    )
                    self.ax.add_patch(arrow)
        
        # Apply highlight effect (subtle for light theme)
        if highlight:
            highlight_glow = CHORD_DIAGRAM_DEFAULTS['highlight_glow']
            # Light background glow
            self.ax.plot(theta_curve, r_curve,
                       color='white', alpha=0.4,
                       linewidth=base_width * highlight_glow,
                       solid_capstyle='round',
                       zorder=zindex - 0.3)
            # Color glow
            self.ax.plot(theta_curve, r_curve,
                       color=color, alpha=0.6,
                       linewidth=base_width * highlight_glow * 0.8,
                       solid_capstyle='round',
                       zorder=zindex - 0.2)
            # Enhanced main line
            self.ax.plot(theta_curve, r_curve,
                       color=color, alpha=min(1.0, alpha * 1.2),  # Reduced for light
                       linewidth=base_width * 1.2,
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
                 color: Union[str, Tuple] = '#E0E0E0', 
                 alpha: Optional[float] = None,
                 glow: bool = False,  # Disabled for light theme
                 border_color: Optional[str] = None) -> Dict[str, float]:
        """
        Add a concentric track to a sector for layered visualizations.
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
            # Darken the color for border
            border_color = self._darken_color(color, 0.3)
        
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
                         shadow: bool = False,  # Changed to False for light theme
                         bg_color: str = 'rgba(255, 255, 255, 0.9)',
                         bg_alpha: float = 0.95) -> None:
        """
        Add readable labels to sectors with light-theme styling.
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
            
            # Use theme text color
            text_color = self.theme_config.text_color
            
            # Add label with light theme styling
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
            
            # Add subtle text shadow for better readability (optional)
            if shadow:
                txt.set_path_effects([
                    patheffects.Stroke(linewidth=1, foreground='white', alpha=0.5),
                    patheffects.Normal()
                ])
            
            # Add background box with rounded corners
            txt.set_bbox(dict(
                boxstyle="round,pad=0.4",
                facecolor=bg_color,
                edgecolor=self.theme_config.panel_border,
                alpha=bg_alpha,
                linewidth=1
            ))
    
    def add_sector_nodes(self, sector_colors: Dict[str, str],
                        node_size_scale: float = 1.0,
                        show_labels: bool = True) -> None:
        """
        Add decorative nodes/circles at sector positions.
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
                size_factor = (degree / max_degree) ** 0.5
                node_size = base_size + (max_size - base_size) * size_factor
            else:
                node_size = base_size
            
            node_size *= node_size_scale
            
            # Calculate proper Cartesian position for node
            x_pos = outer_r * np.cos(angle_rad)
            y_pos = outer_r * np.sin(angle_rad)
            
            # Calculate appropriate radii for node
            node_radius = node_size / 2000
            
            # Create main node circle
            node_color = sector_colors.get(sector, '#666666')
            node_circle = Circle(
                (x_pos, y_pos),
                radius=node_radius,
                facecolor=node_color,
                edgecolor=self.theme_config.node_border_color,
                linewidth=CHORD_DIAGRAM_DEFAULTS['node_border_width'],
                alpha=0.95,
                zorder=10
            )
            self.ax.add_patch(node_circle)
            
            # Add subtle inner highlight for light theme
            highlight_radius = node_radius * 0.7
            highlight_circle = Circle(
                (x_pos, y_pos),
                radius=highlight_radius,
                facecolor='white',
                alpha=0.3,
                zorder=11
            )
            self.ax.add_patch(highlight_circle)
    
    def add_legend(self, title: str = "Link Strength", 
                  position: str = 'right',
                  cmap: Any = None,
                  vmin: float = 0,
                  vmax: float = 1,
                  fontsize: int = 14) -> None:
        """
        Add color legend/bar to the diagram.
        """
        if cmap is None:
            cmap = LightThemeColorPalettes.get_professional_palette()
        
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
        cb.set_label(title, fontsize=fontsize, fontweight='bold', color=self.theme_config.text_color)
        cb.ax.tick_params(labelsize=fontsize-2, colors=self.theme_config.text_color)
        
        # Set background for colorbar
        cb.ax.set_facecolor('none')
    
    def add_statistics_box(self, stats: Dict[str, Any],
                          position: Tuple[float, float] = (0.02, 0.02),
                          fontsize: int = 12) -> None:
        """
        Add statistics text box to the diagram.
        """
        stats_text = []
        for key, value in stats.items():
            if isinstance(value, float):
                stats_text.append(f"{key}: {value:.2f}")
            else:
                stats_text.append(f"{key}: {value}")
        
        stats_str = "\n".join(stats_text)
        
        # Add text box with light theme styling
        self.fig.text(position[0], position[1], stats_str,
                     fontsize=fontsize,
                     fontfamily='monospace',
                     fontweight='bold',
                     color=self.theme_config.text_color,
                     alpha=0.95,
                     bbox=dict(boxstyle='round,pad=0.8',
                             facecolor=self.theme_config.panel_bg,
                             edgecolor=self.theme_config.panel_border,
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
        
        self.ax.set_ylim(0, max_radius * 1.4)
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        
        # Frame visibility
        self.ax.spines['polar'].set_visible(show_frame)
        if show_frame:
            self.ax.spines['polar'].set_color('#666666')
            self.ax.spines['polar'].set_linewidth(1.0)  # Thinner for light theme
        
        # Add grid if requested
        if show_grid:
            self.ax.grid(True, alpha=0.1, color=self.theme_config.grid_color, linewidth=0.5)
        
        # Add title with light theme styling
        if title:
            title_text = self.ax.set_title(
                title, 
                fontsize=28,
                fontweight='bold', 
                pad=40,
                color=self.theme_config.text_color,
                fontfamily='sans-serif',
                alpha=0.98
            )
            
            # Add subtle shadow effect for light theme
            title_text.set_path_effects([
                patheffects.Stroke(linewidth=2, foreground='white', alpha=0.5),
                patheffects.Normal()
            ])
        
        # Remove margins
        self.fig.tight_layout(pad=2.0)
        return self.fig
    
    def _lighten_color(self, color: str, amount: float = 0.5) -> str:
        """Lighten a color by a given amount."""
        try:
            c = mcolors.to_rgb(color)
            c = colorsys.rgb_to_hls(*c)
            new_color = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
            return mcolors.to_hex(new_color)
        except:
            return '#FFFFFF'
    
    def _darken_color(self, color: str, amount: float = 0.5) -> str:
        """Darken a color by a given amount."""
        try:
            c = mcolors.to_rgb(color)
            c = colorsys.rgb_to_hls(*c)
            new_color = colorsys.hls_to_rgb(c[0], c[1] * (1 - amount), c[2])
            return mcolors.to_hex(new_color)
        except:
            return '#000000'

# ============================================================================
# HIGH-LEVEL API - LIGHT THEME NETWORK VISUALIZATION
# ============================================================================

def create_light_theme_chord_diagram(
    data: Any,
    data_type: str = 'matrix',
    figsize: Tuple[int, int] = (24, 24),
    dpi: int = 300,
    title: str = "Network Chord Diagram",
    
    # Theme configuration
    theme: str = 'light',
    theme_config: Optional[ThemeConfig] = None,
    
    # Layout parameters
    start_degree: float = 0,
    direction: str = 'clockwise',
    big_gap: float = 15.0,
    small_gap: float = 2.0,
    sector_order: Optional[List[str]] = None,
    
    # Sector styling
    sector_colors: Optional[Union[str, Dict[str, str]]] = 'light',
    sector_color_palette: str = 'light',
    sector_labels: Optional[Dict[str, str]] = None,
    sector_label_fontsize: int = 18,
    sector_label_fontweight: str = 'bold',
    sector_label_offset: float = 0.30,
    sector_label_shadow: bool = False,
    show_sector_nodes: bool = True,
    sector_node_scale: float = 1.2,
    
    # Link styling
    link_colors: Union[str, Dict[Tuple[str, str], str]] = 'value',
    link_color_palette: str = 'professional',
    link_alpha: float = 0.85,
    link_width_scale: float = 5.0,
    link_min_width: float = 1.5,
    link_max_width: float = 25.0,
    link_glow: bool = True,
    link_glow_intensity: float = 1.5,  # Reduced for light theme
    link_border: bool = True,
    link_border_width: float = 0.8,
    link_border_color: str = '#374151',
    
    # Directional features
    directional: Union[bool, Dict[Tuple[str, str], int]] = False,
    direction_type: Union[str, List[str]] = ['diffHeight', 'arrows'],
    arrow_length: float = 0.18,
    arrow_width: float = 0.10,
    diff_height: float = 0.06,
    
    # Highlighting
    highlight_links: Optional[List[Tuple[str, str]]] = None,
    highlight_color: str = '#DC2626',  # Red for light theme
    highlight_alpha: float = 0.98,
    
    # Advanced features
    symmetric: bool = False,
    reduce_threshold: float = 0.001,
    link_sort: bool = True,
    link_decreasing: bool = True,
    link_zindex: Union[str, List[float]] = 'value',
    link_gradient: bool = False,
    
    # Visual effects
    background_color: str = '#FFFFFF',
    grid_color: str = '#E0E0E0',
    grid_alpha: float = 0.10,
    sector_glow: bool = False,
    sector_glow_width: float = 8,
    sector_glow_alpha: float = 0.3,
    show_frame: bool = False,
    show_grid: bool = False,
    radial_gradient: bool = True,
    
    # Multiple tracks
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
    simplify_curves: bool = False,
    
    # Cache control
    use_cache: bool = True,
    cache_key: Optional[str] = None
) -> plt.Figure:
    """
    Create light-theme chord diagrams with professional styling.
    """
    # Get theme configuration
    if theme_config is None:
        theme_config = ThemeConfig.get_theme(theme)
    
    # Update defaults with theme configuration
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
        'background_color': background_color or theme_config.background_color,
        'grid_color': grid_color or theme_config.grid_color,
        'reduce_threshold': reduce_threshold,
        'sector_glow': sector_glow,
        'sector_glow_width': sector_glow_width,
        'sector_glow_alpha': sector_glow_alpha,
        'node_size_min': 150,
        'node_size_max': 800,
        'link_border_color': link_border_color or theme_config.link_border_color,
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
        groups = {s: 0 for s in all_sectors}
        
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
    # DIAGRAM CONSTRUCTION - LIGHT THEME VERSION
    # ========================================================================
    diagram = LightThemeChordDiagram(figsize=figsize, dpi=dpi, 
                                     background_color=background_color or theme_config.background_color,
                                     theme_config=theme_config)
    diagram.big_gap = big_gap
    diagram.small_gap = small_gap
    diagram.set_start_degree(start_degree)
    diagram.set_direction(direction == 'clockwise')
    
    # Initialize sectors with grouping
    diagram.initialize_sectors(all_sectors, groups)
    
    # Generate sector colors based on palette choice
    if isinstance(sector_colors, str) and sector_colors == 'light':
        sector_color_list = LightThemeColorPalettes.get_sector_colors(
            len(all_sectors), 
            palette=sector_color_palette
        )
        sector_colors = {sector: color for sector, color in zip(all_sectors, sector_color_list)}
    elif isinstance(sector_colors, dict):
        pass  # Use provided colors
    else:
        # Default to light palette
        sector_color_list = LightThemeColorPalettes.get_sector_colors(len(all_sectors), palette='light')
        sector_colors = {sector: color for sector, color in zip(all_sectors, sector_color_list)}
    
    # Add tracks to all sectors
    track_height = 0.20
    for sector in all_sectors:
        sector_color = sector_colors.get(sector, '#666666')
        
        for track_idx in range(tracks):
            track_color = track_colors[track_idx] if track_colors and track_idx < len(track_colors) else theme_config.grid_color
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
    # LINK PROCESSING & STYLING - LIGHT THEME VERSION
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
            if link_color_palette == 'professional':
                link_cmap = LightThemeColorPalettes.get_professional_palette()
            elif link_color_palette == 'pastel':
                link_cmap = LightThemeColorPalettes.get_pastel_palette()
            elif link_color_palette == 'warm':
                link_cmap = LightThemeColorPalettes.get_warm_palette()
            elif link_color_palette == 'cool':
                link_cmap = LightThemeColorPalettes.get_cool_palette()
            elif link_color_palette == 'earth':
                link_cmap = LightThemeColorPalettes.get_earth_palette()
            else:
                link_cmap = plt.cm.Blues  # Default to Blues for light theme
            
            for link in links:
                link_color_map[(link['source'], link['target'])] = link_cmap(norm(link['value']))
        elif link_colors == 'single_color':
            for link in links:
                link_color_map[(link['source'], link['target'])] = '#3B82F6'  # Professional blue
        else:
            for link in links:
                link_color_map[(link['source'], link['target'])] = link_colors
    elif isinstance(link_colors, dict):
        link_color_map = link_colors
    else:
        for link in links:
            link_color_map[(link['source'], link['target'])] = link_colors
    
    # Sort links by value if requested
    if link_sort:
        links.sort(key=lambda x: x['value'], reverse=link_decreasing)
    
    # Assign z-index based on strategy
    if link_zindex == 'value':
        max_val = max(link['value'] for link in links) if links else 1
        for link in links:
            link['zindex'] = 2 + (link['value'] / max_val) * 18
    elif link_zindex == 'random':
        for link in links:
            link['zindex'] = 2 + random.random() * 18
    elif isinstance(link_zindex, (list, np.ndarray)):
        for i, link in enumerate(links):
            link['zindex'] = link_zindex[i] if i < len(link_zindex) else 10.0
    else:
        for link in links:
            link['zindex'] = 10.0
    
    # Create links with light theme styling
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
        
        # Create the link
        try:
            diagram.create_light_link(
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
    # FINALIZATION - LIGHT THEME
    # ========================================================================
    # Add sector labels with light theme styling
    diagram.add_sector_labels(
        sector_labels,
        fontsize=sector_label_fontsize,
        offset=sector_label_offset,
        fontweight=sector_label_fontweight,
        shadow=sector_label_shadow,
        bg_color=theme_config.label_bg_color,
        bg_alpha=0.95
    )
    
    # Add decorative sector nodes
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
        background_color=background_color or theme_config.background_color,
        show_grid=show_grid
    )
    
    return fig

# ============================================================================
# ENHANCED STREAMLIT APPLICATION WITH LIGHT THEME
# ============================================================================

def create_light_theme_streamlit_app():
    """Production-ready Streamlit application for light theme chord diagrams."""
    
    # Initialize session state
    SessionStateManager.initialize_state()
    
    # Page configuration
    st.set_page_config(
        page_title="📊 Network Chord Diagrams",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Get current theme from session state
    current_theme = st.session_state.theme
    theme_config = ThemeConfig.get_theme(current_theme)
    
    # Custom CSS with dynamic theme support
    css = f"""
    <style>
    /* Global styles */
    .main {{
        padding: 1rem 2rem;
        background: {theme_config.background_color};
    }}
    .stApp {{
        background-color: {theme_config.background_color};
    }}
    
    /* Header styling */
    h1 {{
        color: {theme_config.text_color};
        font-weight: 700;
        margin-bottom: 0.5rem;
        border-bottom: 3px solid {theme_config.highlight_color};
        padding-bottom: 0.5rem;
    }}
    .subtitle {{
        color: {theme_config.text_color};
        opacity: 0.8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }}
    
    /* Section dividers */
    .section-divider {{
        height: 1px;
        background: linear-gradient(to right, transparent, {theme_config.panel_border}, transparent);
        margin: 2rem 0;
    }}
    
    /* Cards and containers */
    .stCard {{
        background: {theme_config.panel_bg};
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid {theme_config.panel_border};
        margin-bottom: 1.5rem;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: {theme_config.panel_bg};
        border-right: 1px solid {theme_config.panel_border};
    }}
    
    /* Buttons */
    .stButton > button {{
        background: {theme_config.highlight_color};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        opacity: 0.9;
        transform: translateY(-1px);
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background: {theme_config.panel_bg} !important;
        border: 1px solid {theme_config.panel_border} !important;
        border-radius: 8px !important;
        color: {theme_config.text_color} !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 16px;
        border-bottom: 1px solid {theme_config.panel_border};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {theme_config.text_color};
        opacity: 0.7;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        color: {theme_config.highlight_color} !important;
        opacity: 1 !important;
        border-bottom: 2px solid {theme_config.highlight_color};
    }}
    
    /* Metrics */
    .stMetric {{
        background: {theme_config.panel_bg};
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid {theme_config.panel_border};
    }}
    .stMetric > div {{
        color: {theme_config.text_color} !important;
    }}
    
    /* Selectbox and sliders */
    .stSelectbox > div > div,
    .stSlider > div > div {{
        color: {theme_config.text_color} !important;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        color: {theme_config.text_color};
        opacity: 0.6;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid {theme_config.panel_border};
    }}
    
    /* Performance indicators */
    .performance-badge {{
        background: {theme_config.highlight_color};
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    
    /* Tooltips */
    .tooltip {{
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted {theme_config.text_color};
    }}
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: {theme_config.text_color};
        color: {theme_config.background_color};
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }}
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
    # Header with theme selector
    col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
    with col_header1:
        st.title("📊 Network Chord Diagrams")
        st.markdown(
            f'<p class="subtitle">Professional network visualizations with light theme optimization. '
            f'Perfect for reports, presentations, and analytical dashboards.</p>',
            unsafe_allow_html=True
        )
    
    with col_header2:
        # Theme selector
        theme_options = ["light", "professional", "pastel", "ocean", "dark"]
        theme_display = ["Light", "Professional", "Pastel", "Ocean", "Dark"]
        selected_theme = st.selectbox(
            "Theme",
            options=theme_display,
            index=theme_display.index(current_theme.capitalize()) if current_theme.capitalize() in theme_display else 0,
            key="theme_selector",
            help="Select color theme for the visualization"
        )
        
        # Update session state if theme changed
        theme_map = {display: option.lower() for display, option in zip(theme_display, theme_options)}
        if theme_map[selected_theme] != st.session_state.theme:
            st.session_state.theme = theme_map[selected_theme]
            st.rerun()
    
    with col_header3:
        # Performance indicator
        perf_stats = SessionStateManager.get_performance_stats()
        st.markdown(f'<div class="performance-badge">🔄 {perf_stats["render_count"]} renders</div>', unsafe_allow_html=True)
        st.caption(f"Avg: {perf_stats['avg_render_time']:.1f}s")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown(f"### ⚙️ Configuration")
        
        # Data source selection with caching
        use_example = st.checkbox("Use example dataset", value=True, key="use_example")
        
        if use_example:
            # Load cached example data
            data = CacheManager.load_data("example", is_example=True)
            if data is not None:
                st.success(f"✅ Loaded example dataset ({data.shape[0]}×{data.shape[1]})")
                data_type = "matrix"
        else:
            uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx', 'txt'])
            if uploaded_file:
                # Cache uploaded file data
                file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
                cache_key = f"uploaded_{file_hash}"
                
                if 'uploaded_data' not in st.session_state or st.session_state.get('uploaded_file_hash') != file_hash:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            data = pd.read_csv(uploaded_file)
                        elif uploaded_file.name.endswith('.xlsx'):
                            data = pd.read_excel(uploaded_file)
                        else:
                            data = pd.read_csv(uploaded_file, sep='\t')
                        
                        st.session_state.uploaded_data = data
                        st.session_state.uploaded_file_hash = file_hash
                        st.success(f"✅ Loaded {data.shape[0]}×{data.shape[1]} dataset")
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
                        st.stop()
                else:
                    data = st.session_state.uploaded_data
                
                data_type = st.radio("Data format", ["matrix", "adjacency_list"], horizontal=True)
                if data_type == "adjacency_list":
                    st.info("Expected columns: source, target, value")
            else:
                st.info("Upload your data or use the example dataset")
                st.stop()
        
        # Quick presets for light themes
        st.markdown("---")
        st.markdown("### 🎨 Style Presets")
        preset = st.selectbox("Apply style preset", 
                            ["Default", "Professional", "Pastel", "Ocean", "Minimal", "Colorful"])
        
        if preset != st.session_state.preset:
            st.session_state.preset = preset
            if preset == "Professional":
                st.session_state.update({
                    'sector_color_palette': 'professional',
                    'link_color_palette': 'professional',
                    'background_color': '#FFFFFF',
                    'directional': True,
                    'link_glow': False
                })
            elif preset == "Pastel":
                st.session_state.update({
                    'sector_color_palette': 'pastel',
                    'link_color_palette': 'pastel',
                    'background_color': '#FEFCE8',
                    'directional': False,
                    'sector_glow': False
                })
            elif preset == "Ocean":
                st.session_state.update({
                    'sector_color_palette': 'cool',
                    'link_color_palette': 'cool',
                    'background_color': '#F0F9FF',
                    'directional': True,
                    'link_glow': True
                })
            elif preset == "Minimal":
                st.session_state.update({
                    'sector_color_palette': 'light',
                    'link_color_palette': 'single_color',
                    'background_color': '#FFFFFF',
                    'directional': False,
                    'link_glow': False,
                    'link_border': False
                })
            elif preset == "Colorful":
                st.session_state.update({
                    'sector_color_palette': 'light',
                    'link_color_palette': 'value',
                    'background_color': '#FFFFFF',
                    'directional': True,
                    'link_glow': True
                })
        
        # Performance settings
        st.markdown("---")
        st.markdown("### ⚡ Performance")
        st.session_state.auto_render = st.checkbox("Auto-render on change", value=True)
        st.session_state.high_quality = st.checkbox("High quality rendering", value=True)
        
        if st.button("Clear Cache", use_container_width=True):
            if 'figure_cache' in st.session_state:
                st.session_state.figure_cache = {}
                st.success("Cache cleared!")
    
    # Main content tabs
    tab_viz, tab_data, tab_settings, tab_export, tab_help = st.tabs([
        "📊 Visualization", 
        "📈 Data Explorer", 
        "⚙️ Settings", 
        "💾 Export", 
        "❓ Help"
    ])
    
    # ============================================================================
    # VISUALIZATION TAB
    # ============================================================================
    with tab_viz:
        st.header("Network Visualization")
        
        # Check if we have data
        if 'data' not in locals():
            st.warning("Please load data in the sidebar first.")
            st.stop()
        
        # Parameter configuration in expanders
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### ⚙️ Visualization Controls")
            
            with st.expander("🧭 Layout & Orientation", expanded=False):
                start_degree = st.slider("Starting angle (°)", 0, 359, 0, 15)
                direction = st.selectbox("Direction", ["clockwise", "counter-clockwise"])
                big_gap = st.slider("Group gap (°)", 0, 45, 15, 1)
                small_gap = st.slider("Sector gap (°)", 0, 15, 3, 1)
            
            with st.expander("🎨 Color & Styling", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    sector_color_palette = st.selectbox(
                        "Sector palette",
                        ["light", "pastel", "professional", "warm", "cool", "earth"],
                        index=["light", "pastel", "professional", "warm", "cool", "earth"].index(
                            st.session_state.get('sector_color_palette', 'light')
                        )
                    )
                    link_color_palette = st.selectbox(
                        "Link palette",
                        ["professional", "pastel", "warm", "cool", "earth", "value", "single"],
                        index=["professional", "pastel", "warm", "cool", "earth", "value", "single"].index(
                            st.session_state.get('link_color_palette', 'professional')
                        )
                    )
                with col_b:
                    bg_color = st.color_picker("Background", 
                                              st.session_state.get('background_color', theme_config.background_color))
                    link_alpha = st.slider("Link opacity", 0.3, 1.0, 0.85, 0.05)
                
                show_sector_nodes = st.checkbox("Show sector nodes", value=True)
                sector_node_scale = st.slider("Node size scale", 0.5, 2.0, 1.2, 0.1)
            
            with st.expander("✨ Visual Effects", expanded=False):
                link_glow = st.checkbox("Link glow effect", 
                                       value=st.session_state.get('link_glow', True))
                link_glow_intensity = st.slider("Glow intensity", 0.5, 5.0, 1.5, 0.5)
                
                link_border = st.checkbox("Link borders", 
                                         value=st.session_state.get('link_border', True))
                link_border_width = st.slider("Border width", 0.2, 2.0, 0.8, 0.2)
            
            with st.expander("➡️ Directional Features", expanded=False):
                directional = st.checkbox("Enable directional flows", 
                                         value=st.session_state.get('directional', False))
                
                direction_type = ["diffHeight", "arrows"]
                arrow_size = 1.5
                
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
            
            # Render button
            render_button = st.button("🔄 Render Visualization", type="primary", use_container_width=True)
        
        with col1:
            # Prepare visualization parameters
            viz_params = {
                'data': data,
                'data_type': data_type,
                'figsize': (24, 24) if st.session_state.high_quality else (16, 16),
                'dpi': 300 if st.session_state.high_quality else 150,
                'title': f"Network Analysis • {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                'theme': st.session_state.theme,
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
                'show_sector_nodes': show_sector_nodes,
                'sector_node_scale': sector_node_scale,
                'reduce_threshold': reduce_threshold,
                'max_links': max_links,
                'directional': directional,
                'direction_type': direction_type,
                'arrow_length': 0.12 * arrow_size,
                'arrow_width': 0.07 * arrow_size,
                'symmetric': False,
                'link_gradient': (link_color_palette == 'value'),
                'show_legend': True,
                'show_statistics': True
            }
            
            # Generate hash for current parameters
            current_hash = CacheManager.generate_figure_hash(viz_params)
            
            # Check if we should render
            should_render = (
                render_button or 
                (st.session_state.auto_render and SessionStateManager.should_render(viz_params)) or
                st.session_state.get('figure_hash') != current_hash
            )
            
            if should_render:
                # Update interaction
                SessionStateManager.update_interaction()
                
                # Generate diagram with error handling
                try:
                    start_time = time.time()
                    with st.spinner("Generating visualization..."):
                        fig = create_light_theme_chord_diagram(**viz_params)
                        
                        # Store in session state and cache
                        st.session_state.current_figure = fig
                        st.session_state.viz_params = viz_params
                        st.session_state.figure_hash = current_hash
                        CacheManager.cache_figure(current_hash, fig)
                        
                        # Update performance stats
                        render_time = time.time() - start_time
                        st.session_state.render_count += 1
                        st.session_state.last_render_time = render_time
                        st.session_state.avg_render_time = (
                            (st.session_state.avg_render_time * (st.session_state.render_count - 1) + render_time) / 
                            st.session_state.render_count
                        )
                    
                    # Display the figure
                    st.pyplot(fig, use_container_width=True, clear_figure=True)
                    
                    # Show performance info
                    st.caption(f"Rendered in {render_time:.2f}s | Cache hit: ❌")
                    
                except Exception as e:
                    st.error(f"❌ Rendering error: {str(e)}")
                    with st.expander("Show traceback"):
                        st.code(traceback.format_exc())
            else:
                # Use cached figure
                cached_fig = CacheManager.get_cached_figure(current_hash)
                if cached_fig:
                    st.pyplot(cached_fig, use_container_width=True, clear_figure=True)
                    st.caption(f"Using cached visualization | Cache hit: ✅")
                elif st.session_state.current_figure:
                    st.pyplot(st.session_state.current_figure, use_container_width=True, clear_figure=True)
                    st.caption(f"Using previous visualization")
                else:
                    st.info("Click 'Render Visualization' to generate the diagram")
        
        # Feature summary
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📌 Visualization Features")
        features_cols = st.columns(4)
        with features_cols[0]:
            st.markdown("🎨 **Light Theme**: Optimized for readability")
        with features_cols[1]:
            st.markdown("⚡ **Performance**: Smart caching system")
        with features_cols[2]:
            st.markdown("📊 **Professional**: Publication-ready output")
        with features_cols[3]:
            st.markdown("🔄 **Interactive**: Real-time parameter adjustment")
    
    # ============================================================================
    # DATA EXPLORER TAB
    # ============================================================================
    with tab_data:
        st.header("Dataset Analysis")
        
        if 'data' in locals():
            if data_type == "matrix":
                # Display statistics
                stats_cols = st.columns(5)
                with stats_cols[0]:
                    st.metric("Rows", data.shape[0])
                with stats_cols[1]:
                    st.metric("Columns", data.shape[1])
                with stats_cols[2]:
                    non_zero = np.sum(np.abs(data.values) > 0.01)
                    st.metric("Non-zero", f"{non_zero:,}")
                with stats_cols[3]:
                    density = non_zero / (data.shape[0] * data.shape[1]) * 100
                    st.metric("Density", f"{density:.1f}%")
                with stats_cols[4]:
                    st.metric("Max Value", f"{data.values.max():.3f}")
                
                # Data preview with styling
                st.markdown("#### 🔍 Data Preview")
                styled_data = data.style.background_gradient(
                    cmap='Blues', 
                    axis=None, 
                    vmin=data.values.min(), 
                    vmax=data.values.max()
                ).format("{:.3f}")
                st.dataframe(styled_data, use_container_width=True, height=400)
                
                # Value distribution
                st.markdown("#### 📈 Value Distribution")
                flat_vals = data.values.flatten()
                fig_hist, ax_hist = plt.subplots(figsize=(10, 4), facecolor=theme_config.background_color)
                ax_hist.set_facecolor(theme_config.background_color)
                n, bins, patches = ax_hist.hist(flat_vals, bins=60, 
                                               color=theme_config.highlight_color, alpha=0.7, 
                                               edgecolor=theme_config.panel_border, linewidth=0.5)
                ax_hist.set_xlabel('Value', color=theme_config.text_color, fontsize=12)
                ax_hist.set_ylabel('Frequency', color=theme_config.text_color, fontsize=12)
                ax_hist.set_title('Distribution of Link Values', color=theme_config.text_color, 
                                 fontsize=14, fontweight='bold')
                ax_hist.grid(True, alpha=0.1, color=theme_config.grid_color)
                ax_hist.tick_params(colors=theme_config.text_color)
                st.pyplot(fig_hist, use_container_width=True)
            else:
                # Adjacency list view
                st.markdown(f"**Total Links**: {len(data)}")
                st.markdown(f"**Unique Sources**: {data.iloc[:,0].nunique()}")
                st.markdown(f"**Unique Targets**: {data.iloc[:,1].nunique()}")
                st.dataframe(data.head(50), use_container_width=True)
        else:
            st.warning("No data loaded. Please load data in the sidebar.")
    
    # ============================================================================
    # SETTINGS TAB
    # ============================================================================
    with tab_settings:
        st.header("Application Settings")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            st.markdown("#### 🎛️ Visualization Settings")
            
            # Quality settings
            quality = st.select_slider(
                "Rendering Quality",
                options=["Low", "Medium", "High", "Ultra"],
                value="High" if st.session_state.high_quality else "Medium"
            )
            st.session_state.high_quality = quality in ["High", "Ultra"]
            
            # Auto-render settings
            st.session_state.auto_render = st.checkbox(
                "Auto-render on parameter change",
                value=st.session_state.auto_render,
                help="Automatically re-render visualization when parameters change"
            )
            
            # Cache settings
            cache_size = st.slider(
                "Maximum cache size",
                min_value=1,
                max_value=20,
                value=10,
                help="Maximum number of visualizations to cache"
            )
            
            # Theme settings
            st.markdown("#### 🎨 Theme Settings")
            theme_preview = st.selectbox(
                "Preview theme",
                options=["light", "professional", "pastel", "ocean", "dark"],
                format_func=lambda x: x.capitalize()
            )
            
            # Show theme preview
            preview_config = ThemeConfig.get_theme(theme_preview)
            st.color_picker("Background", preview_config.background_color, disabled=True)
            st.color_picker("Text", preview_config.text_color, disabled=True)
            st.color_picker("Highlight", preview_config.highlight_color, disabled=True)
        
        with col_set2:
            st.markdown("#### 📊 Performance Metrics")
            
            perf_stats = SessionStateManager.get_performance_stats()
            
            # Display metrics
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("Total Renders", perf_stats['render_count'])
                st.metric("Page Views", perf_stats['page_views'])
            with metric_cols[1]:
                st.metric("Avg Render Time", f"{perf_stats['avg_render_time']:.2f}s")
                st.metric("Interactions", perf_stats['interaction_count'])
            
            # Cache info
            st.markdown("#### 💾 Cache Information")
            cache_info = st.session_state.get('figure_cache', {})
            st.metric("Cached Visualizations", len(cache_info))
            
            if cache_info:
                st.write("Cached parameter hashes:")
                for hash_key in list(cache_info.keys())[:5]:  # Show first 5
                    st.code(hash_key[:16] + "...")
            
            # Reset button
            if st.button("Reset All Settings", type="secondary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key not in ['theme', 'page_views']:  # Keep theme and page views
                        del st.session_state[key]
                st.success("Settings reset! Refreshing page...")
                st.rerun()
    
    # ============================================================================
    # EXPORT TAB
    # ============================================================================
    with tab_export:
        st.header("Export Options")
        
        if 'current_figure' not in st.session_state:
            st.info("💡 Generate a visualization first, then export it here")
        else:
            fig = st.session_state.current_figure
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                st.markdown("#### 🖼️ Image Formats")
                export_dpi = st.slider("Resolution (DPI)", 150, 600, 300, 50)
                
                # PNG Export
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format='png', dpi=export_dpi, 
                          bbox_inches='tight', facecolor=theme_config.background_color, 
                          pad_inches=0.5)
                buf_png.seek(0)
                st.download_button(
                    label="⬇️ Download PNG",
                    data=buf_png,
                    file_name=f"network_chord_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    key="png_download",
                    use_container_width=True
                )
                
                # JPEG Export
                buf_jpeg = io.BytesIO()
                fig.savefig(buf_jpeg, format='jpg', dpi=export_dpi,
                          bbox_inches='tight', facecolor=theme_config.background_color,
                          pad_inches=0.5, quality=95)
                buf_jpeg.seek(0)
                st.download_button(
                    label="⬇️ Download JPEG",
                    data=buf_jpeg,
                    file_name=f"network_chord_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    mime="image/jpeg",
                    key="jpeg_download",
                    use_container_width=True
                )
            
            with col_exp2:
                st.markdown("#### 📄 Vector Formats")
                
                # PDF Export
                buf_pdf = io.BytesIO()
                fig.savefig(buf_pdf, format='pdf', bbox_inches='tight',
                          facecolor=theme_config.background_color, pad_inches=0.5)
                buf_pdf.seek(0)
                st.download_button(
                    label="⬇️ Download PDF",
                    data=buf_pdf,
                    file_name=f"network_chord_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="pdf_download",
                    use_container_width=True
                )
                
                # SVG Export
                buf_svg = io.BytesIO()
                fig.savefig(buf_svg, format='svg', bbox_inches='tight',
                          facecolor=theme_config.background_color, pad_inches=0.5)
                buf_svg.seek(0)
                st.download_button(
                    label="⬇️ Download SVG",
                    data=buf_svg,
                    file_name=f"network_chord_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                    mime="image/svg+xml",
                    key="svg_download",
                    use_container_width=True
                )
            
            # Export settings
            st.markdown("---")
            st.markdown("#### ⚙️ Export Settings")
            
            col_exp3, col_exp4 = st.columns(2)
            with col_exp3:
                include_stats = st.checkbox("Include statistics", value=True)
                include_legend = st.checkbox("Include legend", value=True)
            with col_exp4:
                transparent_bg = st.checkbox("Transparent background", value=False)
                crop_to_content = st.checkbox("Crop to content", value=True)
            
            # Batch export (if multiple visualizations cached)
            if 'figure_cache' in st.session_state and len(st.session_state.figure_cache) > 1:
                st.markdown("---")
                st.markdown("#### 📦 Batch Export")
                if st.button("Export All Cached Visualizations", use_container_width=True):
                    st.info("Batch export functionality would be implemented here")
    
    # ============================================================================
    # HELP TAB
    # ============================================================================
    with tab_help:
        st.header("User Guide & Documentation")
        
        # Create expandable sections
        with st.expander("📖 Getting Started", expanded=True):
            st.markdown("""
            ### Quick Start Guide
            
            1. **Load Data**: Use the sidebar to load your dataset or use the example data
            2. **Configure Visualization**: Adjust parameters in the Visualization tab
            3. **Render**: Click "Render Visualization" or enable auto-render
            4. **Export**: Download your visualization in various formats
            
            ### Data Formats
            
            **Matrix Format** (Recommended):
            - Rows represent sources, columns represent targets
            - Values represent connection strength (0-1 or absolute values)
            - Diagonal values are typically ignored
            
            **Adjacency List Format**:
            - Three columns: source, target, value
            - Each row represents a single connection
            - More flexible for sparse networks
            """)
        
        with st.expander("🎨 Visualization Parameters"):
            st.markdown("""
            ### Key Parameters
            
            **Layout & Orientation**:
            - **Starting Angle**: Where the first sector appears (0° = top)
            - **Direction**: Clockwise or counter-clockwise sector arrangement
            - **Gaps**: Control spacing between sectors and groups
            
            **Color & Styling**:
            - **Sector Palette**: Color scheme for different sectors
            - **Link Palette**: How links are colored (by value, group, etc.)
            - **Opacity**: Control transparency of links
            - **Node Size**: Scale factor for sector nodes
            
            **Visual Effects**:
            - **Glow Effects**: Subtle glow around links (adjustable intensity)
            - **Borders**: Add borders to links for better contrast
            - **Directional Indicators**: Show flow direction with arrows
            
            **Performance**:
            - **Min Link Value**: Filter out weak connections
            - **Max Links**: Limit total number of links displayed
            """)
        
        with st.expander("⚡ Performance Tips"):
            st.markdown("""
            ### Optimization Strategies
            
            **For Large Networks**:
            1. Increase the "Min link value" threshold
            2. Reduce "Max links to display"
            3. Use lower quality settings for faster previews
            
            **Caching System**:
            - The application caches visualizations based on parameters
            - Identical parameter sets use cached results
            - Cache is cleared when parameters change
            
            **Memory Management**:
            - Limit cache size in Settings tab
            - Use "Clear Cache" button if experiencing issues
            - Restart application if memory usage is high
            """)
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            ### Common Issues
            
            **Slow Rendering**:
            - Reduce canvas size or DPI
            - Filter out weak connections
            - Disable visual effects temporarily
            
            **Memory Issues**:
            - Clear the cache
            - Restart the application
            - Reduce cache size in settings
            
            **Visual Artifacts**:
            - Check for NaN or infinite values in data
            - Adjust color scales
            - Try different themes
            
            **Export Problems**:
            - Ensure sufficient disk space
            - Try different formats (PNG vs PDF)
            - Reduce DPI for large files
            """)
        
        with st.expander("📚 Advanced Features"):
            st.markdown("""
            ### Advanced Configuration
            
            **Theme System**:
            - Multiple built-in themes (Light, Professional, Pastel, Ocean, Dark)
            - Custom theme support via configuration
            - Dynamic CSS styling
            
            **Session Management**:
            - Persistent settings across sessions
            - Performance tracking and analytics
            - State preservation on page refresh
            
            **Batch Processing**:
            - Export multiple visualizations
            - Parameter sweeps for comparison
            - Automated report generation
            
            **Integration**:
            - REST API for programmatic access
            - Database connectivity
            - Cloud storage integration
            """)
    
    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    footer_cols = st.columns(3)
    with footer_cols[0]:
        st.markdown(f"**Theme**: {current_theme.capitalize()}")
    with footer_cols[1]:
        st.markdown(f"**Version**: 2.0.0")
    with footer_cols[2]:
        st.markdown(f"**Last Render**: {st.session_state.get('last_render_time', 0):.2f}s")
    
    st.markdown("""
    <div class="footer">
        <p>Network Chord Diagrams • Professional Light Theme • Smart Caching System</p>
        <p>💡 Tip: Use the auto-render feature for real-time updates, and leverage the caching system for optimal performance.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Application entry point."""
    try:
        create_light_theme_streamlit_app()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
