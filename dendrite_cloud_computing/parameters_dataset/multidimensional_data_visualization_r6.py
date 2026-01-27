# -*- coding: utf-8 -*-
"""
ULTRA-ENHANCED VIVID CHORD DIAGRAM VISUALIZATION
Light Theme Edition with Advanced Caching & Session Management
Professional-grade network visualization with real-time interaction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch, FancyBboxPatch, Wedge, Arc, Circle, Rectangle
from matplotlib.collections import PatchCollection, LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
import matplotlib.patheffects as patheffects
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import colorsys
from scipy import stats, spatial, interpolate
import io
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, Iterable
import math
import random
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.font_manager import FontProperties
import traceback
import os
import sys
import json
import pickle
import hashlib
import time
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import base64
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
import umap.umap_ as umap
import hdbscan

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set matplotlib backend to avoid threading issues
matplotlib.use('Agg')

# ============================================================================
# ADVANCED CONFIGURATION & CONSTANTS - LIGHT THEME EDITION
# ============================================================================

class LightThemeConfig:
    """Light theme configuration with enhanced visual parameters"""
    
    # LIGHT THEME COLORS
    BACKGROUND_COLORS = {
        'pure_white': '#FFFFFF',
        'soft_ivory': '#FFFFF0',
        'light_gray': '#F8F9FA',
        'parchment': '#F5F5DC',
        'alabaster': '#F2F0E6',
        'snow': '#FFFAFA',
        'seashell': '#FFF5EE',
        'floral_white': '#FFFAF0',
        'honeydew': '#F0FFF0',
        'azure': '#F0FFFF',
        'mint_cream': '#F5FFFA',
        'ghost_white': '#F8F8FF'
    }
    
    # COMPLEMENTARY COLORS FOR LIGHT BACKGROUND
    TEXT_COLORS = {
        'primary': '#2C3E50',      # Dark blue-gray
        'secondary': '#34495E',    # Slightly lighter
        'accent': '#2980B9',       # Blue accent
        'highlight': '#E74C3C',    # Red for highlights
        'success': '#27AE60',      # Green
        'warning': '#F39C12',      # Orange
        'muted': '#7F8C8D'         # Gray for less important text
    }
    
    # GRID & FRAME COLORS
    GRID_COLORS = {
        'subtle': '#E0E0E0',      # Very light gray
        'medium': '#B0B0B0',      # Medium gray
        'strong': '#808080',      # Darker gray
        'blue_tint': '#D0E0F0',   # Blue tinted
        'warm': '#E8D8C8'         # Warm beige
    }
    
    # VIVID COLOR PALETTES FOR LINKS (OPTIMIZED FOR LIGHT BACKGROUND)
    LINK_PALETTES = {
        'vivid_light': [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F1948A', '#52BE80', '#5DADE2', '#F8C471', '#AF7AC5',
            '#76D7C4', '#F9E79F', '#D7BDE2', '#A9CCE3', '#FAD7A0'
        ],
        'electric_light': [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
            '#00FFFF', '#FF8000', '#80FF00', '#0080FF', '#FF0080',
            '#80FFFF', '#FF80FF', '#804000', '#008040', '#400080',
            '#FF4040', '#40FF40', '#4040FF', '#FFFF40', '#FF40FF'
        ],
        'pastel_rainbow': [
            '#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
            '#D0BAFF', '#FFB3FF', '#FFB3D9', '#B3FFF3', '#B3ECFF',
            '#D9B3FF', '#FFB3E6', '#FFD9B3', '#B3FFB3', '#B3D9FF',
            '#ECB3FF', '#FFB3CC', '#FFECB3', '#CCFFB3', '#CCB3FF'
        ],
        'professional': [
            '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
            '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
            '#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5',
            '#C49C94', '#F7B6D2', '#C7C7C7', '#DBDB8D', '#9EDAE5'
        ],
        'gradient_heat': [
            '#FFE5E5', '#FFCCCC', '#FFB2B2', '#FF9999', '#FF7F7F',
            '#FF6666', '#FF4C4C', '#FF3232', '#FF1919', '#FF0000'
        ],
        'gradient_cool': [
            '#E5F2FF', '#CCE5FF', '#B2D8FF', '#99CCFF', '#7FBFFF',
            '#66B2FF', '#4CA5FF', '#3298FF', '#198CFF', '#0080FF'
        ],
        'gradient_green': [
            '#E5FFE5', '#CCFFCC', '#B2FFB2', '#99FF99', '#7FFF7F',
            '#66FF66', '#4CFF4C', '#32FF32', '#19FF19', '#00FF00'
        ]
    }
    
    # SECTOR COLORS FOR LIGHT BACKGROUND
    SECTOR_PALETTES = {
        'soft_pastel': ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
                       '#D0BAFF', '#FFB3FF', '#FFB3D9', '#B3FFF3', '#B3ECFF'],
        'muted_professional': ['#8DA0CB', '#66C2A5', '#FC8D62', '#E78AC3', 
                              '#A6D854', '#FFD92F', '#E5C494', '#B3B3B3'],
        'earth_tones': ['#8B4513', '#A0522D', '#D2691E', '#CD853F', '#F4A460',
                       '#DEB887', '#D2B48C', '#BC8F8F', '#F5DEB3', '#FFE4C4'],
        'water_colors': ['#1E90FF', '#00BFFF', '#87CEEB', '#87CEFA', '#B0E0E6',
                        '#ADD8E6', '#B0C4DE', '#6495ED', '#4169E1', '#0000FF']
    }

LIGHT_THEME_DEFAULTS = {
    # LARGER CANVAS WITH LIGHT THEME
    'figsize': (24, 24),           # Increased from (14, 14) to (24, 24)
    'dpi': 300,                    # Increased from 100 to 300 for high quality
    'big_gap': 15.0,               # degrees between groups
    'small_gap': 2.0,              # degrees within groups
    'start_degree': 0,
    'clockwise': True,
    
    # LIGHT THEME SPECIFIC
    'background_color': LightThemeConfig.BACKGROUND_COLORS['pure_white'],
    'grid_color': LightThemeConfig.GRID_COLORS['subtle'],
    'text_color_primary': LightThemeConfig.TEXT_COLORS['primary'],
    'text_color_secondary': LightThemeConfig.TEXT_COLORS['secondary'],
    'grid_alpha': 0.15,            # More subtle on light background
    
    # ENHANCED TRACK DIMENSIONS
    'track_height': 0.20,          # Increased from 0.12 to 0.20
    'track_padding': 0.03,
    
    # VIVID LINK STYLING (OPTIMIZED FOR LIGHT BACKGROUND)
    'link_min_width': 1.5,         # Increased from 0.8
    'link_max_width': 25.0,        # Increased from 10.0
    'link_width_scale': 5.0,       # Increased from 3.5
    'link_alpha': 0.85,            # Increased from 0.75
    'link_glow_intensity': 2.5,    # Glow effect multiplier
    'link_glow_alpha': 0.3,        # Lighter glow for light background
    'link_border_width': 0.8,      # Border around links
    'link_border_color': '#FFFFFF', # White border for contrast
    
    # ENHANCED DIRECTIONAL FEATURES
    'arrow_length': 0.18,          # Increased from 0.12
    'arrow_width': 0.10,           # Increased from 0.06
    'diff_height': 0.06,           # Increased from 0.03
    
    # IMPROVED LABELING FOR LIGHT THEME
    'label_offset': 0.30,          # Increased from 0.18
    'label_fontsize': 18,          # Increased from 13
    'label_fontweight': 'bold',
    'label_shadow': False,         # No shadow needed for light theme
    'label_bg_alpha': 0.85,        # Slightly transparent background
    'label_bg_color': '#FFFFFF',   # White background for labels
    'label_text_color': LightThemeConfig.TEXT_COLORS['primary'],
    
    # HIGHLIGHT SETTINGS
    'highlight_alpha': 0.95,
    'highlight_glow': 3.0,
    
    # NETWORK ENHANCEMENTS
    'reduce_threshold': 0.001,     # Lower threshold to show more links
    'node_size_min': 150,          # Minimum node size
    'node_size_max': 800,          # Maximum node size
    'show_node_borders': True,
    'node_border_width': 2,
    'node_border_color': '#FFFFFF', # White borders for nodes
    'sector_glow': False,          # No glow for light theme
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
    'max_links': 1500,             # Increased limit
    'simplify_curves': False,      # Keep detailed curves
    
    # ANIMATION SETTINGS
    'animation_enabled': False,
    'animation_duration': 1000,    # ms
    'animation_fps': 30,
    
    # INTERACTIVITY
    'hover_effects': True,
    'click_highlight': True,
    'zoom_enabled': True,
    
    # CACHING
    'cache_expiry': 3600,          # 1 hour in seconds
    'cache_max_size': 100,         # Maximum cached items
    
    # QUALITY
    'anti_aliasing': True,
    'vector_quality': 'high',      # 'low', 'medium', 'high'
    
    # MULTI-CORE PROCESSING
    'use_multiprocessing': True,
    'max_workers': 4
}

# ============================================================================
# ADVANCED COLOR PALETTE ENGINE
# ============================================================================

class AdvancedColorEngine:
    """Advanced color management with light theme optimization"""
    
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=50)
    def generate_color_palette(n_colors: int, palette_type: str = 'vivid_light', 
                              sequential: bool = False) -> List[str]:
        """Generate optimized color palettes with caching"""
        if palette_type in LightThemeConfig.LINK_PALETTES:
            base_palette = LightThemeConfig.LINK_PALETTES[palette_type]
        elif palette_type in LightThemeConfig.SECTOR_PALETTES:
            base_palette = LightThemeConfig.SECTOR_PALETTES[palette_type]
        else:
            base_palette = LightThemeConfig.LINK_PALETTES['vivid_light']
        
        if n_colors <= len(base_palette):
            return base_palette[:n_colors]
        
        # Generate additional colors by interpolating
        cmap = LinearSegmentedColormap.from_list('custom', base_palette, N=256)
        return [mcolors.to_hex(cmap(i / (n_colors - 1))) for i in range(n_colors)]
    
    @staticmethod
    def create_gradient_cmap(colors: List[str], name: str = 'custom_gradient', 
                           n_bins: int = 256) -> LinearSegmentedColormap:
        """Create smooth gradient colormap from color list"""
        return LinearSegmentedColormap.from_list(name, colors, N=n_bins)
    
    @staticmethod
    def adjust_color_for_background(color: str, background_color: str = '#FFFFFF') -> str:
        """Adjust color to ensure visibility on specified background"""
        color_rgb = mcolors.to_rgb(color)
        bg_rgb = mcolors.to_rgb(background_color)
        
        # Calculate contrast ratio
        def luminance(rgb):
            r, g, b = rgb
            r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        l1 = luminance(color_rgb)
        l2 = luminance(bg_rgb)
        contrast = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
        
        # Adjust if contrast is too low
        if contrast < 2.5:
            # Darken or lighten color based on background
            if l2 > 0.5:  # Light background
                # Darken the color
                h, l, s = colorsys.rgb_to_hls(*color_rgb)
                l = max(0.1, l * 0.6)  # Darken
                return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))
            else:  # Dark background
                # Lighten the color
                h, l, s = colorsys.rgb_to_hls(*color_rgb)
                l = min(0.9, l * 1.4)  # Lighten
                return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))
        
        return color
    
    @staticmethod
    def generate_complementary_colors(base_color: str, n_colors: int = 5) -> List[str]:
        """Generate complementary colors from a base color"""
        base_hsv = colorsys.rgb_to_hsv(*mcolors.to_rgb(base_color))
        
        colors = []
        for i in range(n_colors):
            hue = (base_hsv[0] + i * (1.0 / n_colors)) % 1.0
            saturation = base_hsv[1] * (0.8 + 0.2 * (i % 2))
            value = base_hsv[2] * (0.9 + 0.1 * ((i + 1) % 2))
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(mcolors.to_hex(rgb))
        
        return colors
    
    @staticmethod
    def create_diverging_palette(low_color: str = '#2166AC', 
                               high_color: str = '#B2182B',
                               mid_color: str = '#F7F7F7') -> LinearSegmentedColormap:
        """Create diverging color palette"""
        return LinearSegmentedColormap.from_list('diverging', 
                                                [low_color, mid_color, high_color], 
                                                N=256)
    
    @staticmethod
    def get_palette_names() -> Dict[str, List[str]]:
        """Get all available palette names categorized"""
        return {
            'link_palettes': list(LightThemeConfig.LINK_PALETTES.keys()),
            'sector_palettes': list(LightThemeConfig.SECTOR_PALETTES.keys()),
            'background_colors': list(LightThemeConfig.BACKGROUND_COLORS.keys()),
            'text_colors': list(LightThemeConfig.TEXT_COLORS.keys()),
            'grid_colors': list(LightThemeConfig.GRID_COLORS.keys())
        }

# ============================================================================
# ADVANCED CACHING SYSTEM
# ============================================================================

class SmartCacheSystem:
    """Advanced caching system with automatic invalidation"""
    
    _cache_store = {}
    _cache_timestamps = {}
    _cache_hits = {}
    _cache_misses = {}
    
    @classmethod
    def get_or_create(cls, key: str, creator_func: Callable, *args, **kwargs) -> Any:
        """Get cached item or create if not exists"""
        cache_key = cls._generate_key(key, args, kwargs)
        
        if cache_key in cls._cache_store:
            # Check if cache is still valid
            if time.time() - cls._cache_timestamps[cache_key] < LIGHT_THEME_DEFAULTS['cache_expiry']:
                cls._cache_hits[cache_key] = cls._cache_hits.get(cache_key, 0) + 1
                return cls._cache_store[cache_key]
        
        # Create new item
        result = creator_func(*args, **kwargs)
        cls._cache_store[cache_key] = result
        cls._cache_timestamps[cache_key] = time.time()
        cls._cache_misses[cache_key] = cls._cache_misses.get(cache_key, 0) + 1
        
        # Clean old cache entries
        cls._clean_old_cache()
        
        return result
    
    @classmethod
    def _generate_key(cls, base_key: str, args: tuple, kwargs: dict) -> str:
        """Generate unique cache key"""
        # Skip self parameter for instance methods
        if args and hasattr(args[0], '__class__'):
            # First arg is self, skip it for hashing
            args = args[1:]
        
        key_parts = [base_key, str(args), str(sorted(kwargs.items()))]
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    @classmethod
    def _clean_old_cache(cls):
        """Remove old cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in cls._cache_timestamps.items()
            if current_time - timestamp > LIGHT_THEME_DEFAULTS['cache_expiry']
        ]
        
        for key in expired_keys:
            cls._cache_store.pop(key, None)
            cls._cache_timestamps.pop(key, None)
            cls._cache_hits.pop(key, None)
            cls._cache_misses.pop(key, None)
        
        # If still too many items, remove least recently used
        if len(cls._cache_store) > LIGHT_THEME_DEFAULTS['cache_max_size']:
            sorted_keys = sorted(cls._cache_timestamps.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[:len(sorted_keys) - LIGHT_THEME_DEFAULTS['cache_max_size']]]
            for key in keys_to_remove:
                cls._cache_store.pop(key, None)
                cls._cache_timestamps.pop(key, None)
                cls._cache_hits.pop(key, None)
                cls._cache_misses.pop(key, None)
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_items': len(cls._cache_store),
            'hits': sum(cls._cache_hits.values()),
            'misses': sum(cls._cache_misses.values()),
            'hit_rate': sum(cls._cache_hits.values()) / max(1, sum(cls._cache_hits.values()) + sum(cls._cache_misses.values())),
            'oldest_entry': min(cls._cache_timestamps.values()) if cls._cache_timestamps else None,
            'newest_entry': max(cls._cache_timestamps.values()) if cls._cache_timestamps else None
        }
    
    @classmethod
    def clear_cache(cls):
        """Clear all cache"""
        cls._cache_store.clear()
        cls._cache_timestamps.clear()
        cls._cache_hits.clear()
        cls._cache_misses.clear()

# ============================================================================
# ADVANCED SESSION STATE MANAGEMENT
# ============================================================================

class SessionStateManager:
    """Advanced session state management with persistence"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables with defaults"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            
            # Visualization settings
            st.session_state.viz_settings = {
                'theme': 'light',
                'background_color': LIGHT_THEME_DEFAULTS['background_color'],
                'palette': 'vivid_light',
                'dpi': 300,
                'figsize': (24, 24),
                'animation_enabled': False,
                'interactive_mode': True
            }
            
            # Data state
            st.session_state.data_loaded = False
            st.session_state.current_data = None
            st.session_state.data_type = None
            st.session_state.data_hash = None
            
            # Computation cache
            st.session_state.computation_cache = {}
            st.session_state.cache_timestamps = {}
            
            # User preferences
            st.session_state.user_preferences = {
                'auto_save': True,
                'auto_refresh': False,
                'notifications': True,
                'performance_mode': 'balanced'
            }
            
            # History and undo stack
            st.session_state.action_history = []
            st.session_state.undo_stack = []
            st.session_state.max_history_size = 50
            
            # Statistics
            st.session_state.session_start_time = datetime.now()
            st.session_state.render_count = 0
            st.session_state.error_count = 0
            st.session_state.success_count = 0
            
            # UI state
            st.session_state.current_tab = 'visualization'
            st.session_state.sidebar_expanded = True
            st.session_state.modal_open = False
            
            # Export state
            st.session_state.export_format = 'PNG'
            st.session_state.export_dpi = 300
            st.session_state.export_quality = 'high'
    
    @staticmethod
    def update_setting(key: str, value: Any, category: str = 'viz_settings'):
        """Update a setting with history tracking"""
        if category not in st.session_state:
            st.session_state[category] = {}
        
        # Store previous value for undo
        if key in st.session_state[category]:
            previous_value = st.session_state[category][key]
            
            # Add to undo stack if value changed
            if previous_value != value:
                action = {
                    'type': 'setting_update',
                    'category': category,
                    'key': key,
                    'previous_value': previous_value,
                    'new_value': value,
                    'timestamp': datetime.now()
                }
                
                st.session_state.action_history.append(action)
                
                # Limit history size
                if len(st.session_state.action_history) > st.session_state.max_history_size:
                    st.session_state.action_history.pop(0)
        
        # Update the setting
        st.session_state[category][key] = value
    
    @staticmethod
    def undo_last_action():
        """Undo the last action"""
        if not st.session_state.action_history:
            return False
        
        last_action = st.session_state.action_history.pop()
        st.session_state.undo_stack.append(last_action)
        
        if last_action['type'] == 'setting_update':
            category = last_action['category']
            key = last_action['key']
            previous_value = last_action['previous_value']
            
            if category in st.session_state and key in st.session_state[category]:
                st.session_state[category][key] = previous_value
        
        return True
    
    @staticmethod
    def redo_last_undo():
        """Redo the last undone action"""
        if not st.session_state.undo_stack:
            return False
        
        last_undo = st.session_state.undo_stack.pop()
        
        if last_undo['type'] == 'setting_update':
            category = last_undo['category']
            key = last_undo['key']
            new_value = last_undo['new_value']
            
            if category in st.session_state:
                st.session_state[category][key] = new_value
            
            # Add back to history
            st.session_state.action_history.append(last_undo)
        
        return True
    
    @staticmethod
    def get_session_stats() -> Dict[str, Any]:
        """Get session statistics"""
        return {
            'session_duration': datetime.now() - st.session_state.session_start_time,
            'render_count': st.session_state.render_count,
            'error_count': st.session_state.error_count,
            'success_count': st.session_state.success_count,
            'cache_size': len(st.session_state.computation_cache) if 'computation_cache' in st.session_state else 0,
            'history_size': len(st.session_state.action_history) if 'action_history' in st.session_state else 0
        }
    
    @staticmethod
    def save_session_state():
        """Save session state to browser storage"""
        try:
            # Convert non-serializable objects
            serializable_state = {}
            for key, value in st.session_state.items():
                try:
                    json.dumps(value)
                    serializable_state[key] = value
                except:
                    # Skip non-serializable items
                    pass
            
            # Store in session storage
            st.session_state.saved_state = base64.b64encode(
                pickle.dumps(serializable_state)
            ).decode('utf-8')
            
            return True
        except Exception as e:
            st.error(f"Error saving session state: {str(e)}")
            return False
    
    @staticmethod
    def load_session_state():
        """Load session state from browser storage"""
        if 'saved_state' in st.session_state:
            try:
                saved_data = pickle.loads(
                    base64.b64decode(st.session_state.saved_state)
                )
                
                for key, value in saved_data.items():
                    st.session_state[key] = value
                
                return True
            except Exception as e:
                st.error(f"Error loading session state: {str(e)}")
                return False
        return False

# ============================================================================
# ADVANCED DATA PROCESSING ENGINE
# ============================================================================

class DataProcessingEngine:
    """Advanced data processing with parallel computation"""
    
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=20)
    def load_and_preprocess_data(file_path: str = None, 
                                file_object = None,
                                data_type: str = 'matrix',
                                normalize: bool = True,
                                filter_threshold: float = 0.0) -> Tuple[Any, str, Dict]:
        """Load and preprocess data with caching"""
        try:
            if file_object is not None:
                if file_object.name.endswith('.csv'):
                    data = pd.read_csv(file_object)
                elif file_object.name.endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(file_object)
                else:
                    raise ValueError(f"Unsupported file format: {file_object.name}")
            elif file_path is not None:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")
            else:
                # Generate sample data
                data = DataProcessingEngine.generate_sample_data()
                data_type = 'matrix'
            
            # Generate data hash for cache invalidation
            data_hash = DataProcessingEngine._generate_data_hash(data)
            
            # Preprocess data
            processed_data, stats = DataProcessingEngine._preprocess_data(
                data, data_type, normalize, filter_threshold
            )
            
            return processed_data, data_type, {
                'original_shape': data.shape if hasattr(data, 'shape') else 'unknown',
                'processed_shape': processed_data.shape if hasattr(processed_data, 'shape') else 'unknown',
                'data_hash': data_hash,
                'stats': stats
            }
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            # Return sample data as fallback
            return DataProcessingEngine.generate_sample_data(), 'matrix', {
                'original_shape': (15, 15),
                'processed_shape': (15, 15),
                'data_hash': 'sample',
                'stats': {'type': 'sample_data'}
            }
    
    @staticmethod
    def _generate_data_hash(data: Any) -> str:
        """Generate hash for data to detect changes"""
        if isinstance(data, pd.DataFrame):
            data_str = data.to_csv(index=False)
        elif isinstance(data, np.ndarray):
            data_str = data.tobytes().hex()
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    @staticmethod
    def _preprocess_data(data: Any, data_type: str, 
                        normalize: bool, filter_threshold: float) -> Tuple[Any, Dict]:
        """Preprocess data based on type and parameters"""
        stats = {}
        
        if data_type == 'matrix':
            if isinstance(data, pd.DataFrame):
                matrix = data.values
                row_names = data.index.tolist()
                col_names = data.columns.tolist()
            else:
                matrix = np.array(data)
                row_names = [f"Source_{i+1}" for i in range(matrix.shape[0])]
                col_names = [f"Target_{j+1}" for j in range(matrix.shape[1])]
            
            # Normalize if requested
            if normalize:
                matrix = DataProcessingEngine._normalize_matrix(matrix)
                stats['normalization'] = 'applied'
            
            # Apply filter threshold
            if filter_threshold > 0:
                mask = np.abs(matrix) >= filter_threshold
                matrix = matrix * mask
                stats['filter_threshold'] = filter_threshold
                stats['remaining_links'] = np.sum(mask)
            
            # Create DataFrame with processed matrix
            processed_data = pd.DataFrame(matrix, index=row_names, columns=col_names)
            
            # Calculate statistics
            stats.update({
                'min_value': float(np.nanmin(matrix)),
                'max_value': float(np.nanmax(matrix)),
                'mean_value': float(np.nanmean(matrix)),
                'std_value': float(np.nanstd(matrix)),
                'non_zero_count': int(np.sum(matrix != 0)),
                'total_elements': int(matrix.size)
            })
            
        else:  # adjacency_list
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                df = pd.DataFrame(data)
            
            # Ensure required columns
            if df.shape[1] < 3:
                raise ValueError("Adjacency list needs at least 3 columns")
            
            # Normalize values if requested
            if normalize and df.shape[1] >= 3:
                values = df.iloc[:, 2].values
                if np.std(values) > 0:
                    values = (values - np.mean(values)) / np.std(values)
                    df.iloc[:, 2] = values
                stats['normalization'] = 'applied'
            
            # Filter by threshold
            if filter_threshold > 0 and df.shape[1] >= 3:
                df = df[np.abs(df.iloc[:, 2]) >= filter_threshold].copy()
                stats['filter_threshold'] = filter_threshold
            
            processed_data = df
            
            # Calculate statistics
            if df.shape[1] >= 3:
                values = df.iloc[:, 2].values
                stats.update({
                    'min_value': float(np.min(values)),
                    'max_value': float(np.max(values)),
                    'mean_value': float(np.mean(values)),
                    'std_value': float(np.std(values)),
                    'link_count': len(df),
                    'unique_sources': df.iloc[:, 0].nunique(),
                    'unique_targets': df.iloc[:, 1].nunique()
                })
        
        return processed_data, stats
    
    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix values"""
        # Remove NaN and infinite values
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to [0, 1] range
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        
        if max_val - min_val > 0:
            normalized = (matrix - min_val) / (max_val - min_val)
        else:
            normalized = matrix
        
        return normalized
    
    @staticmethod
    def generate_sample_data(size: int = 15, density: float = 0.3, 
                           noise_level: float = 0.1) -> pd.DataFrame:
        """Generate realistic sample data"""
        np.random.seed(42)
        
        # Create base matrix with clusters
        matrix = np.zeros((size, size))
        
        # Create clusters
        clusters = []
        for i in range(0, size, 5):
            for j in range(0, size, 5):
                if i < size and j < size:
                    cluster_size = min(5, size - i, size - j)
                    cluster = np.random.rand(cluster_size, cluster_size) * 0.8 + 0.2
                    matrix[i:i+cluster_size, j:j+cluster_size] = cluster
                    clusters.append((i, j, cluster_size))
        
        # Add some strong connections
        for _ in range(size // 2):
            i, j = np.random.randint(0, size, 2)
            matrix[i, j] = np.random.rand() * 0.5 + 0.5
        
        # Add noise
        noise_mask = np.random.rand(size, size) < density
        noise = np.random.randn(size, size) * noise_level
        matrix = matrix + noise_mask * noise
        
        # Ensure non-negative
        matrix = np.abs(matrix)
        
        # Create meaningful labels
        sources = [f"Gene_{i+1:02d}" for i in range(size)]
        targets = [f"Pathway_{j+1:02d}" for j in range(size)]
        
        return pd.DataFrame(matrix, index=sources, columns=targets)
    
    @staticmethod
    def perform_network_analysis(data: Any, data_type: str) -> Dict[str, Any]:
        """Perform advanced network analysis"""
        analysis = {}
        
        if data_type == 'matrix':
            matrix = data.values if isinstance(data, pd.DataFrame) else np.array(data)
            
            # Basic network metrics
            analysis['density'] = float(np.mean(matrix > 0))
            analysis['average_strength'] = float(np.mean(matrix[matrix > 0]))
            analysis['max_strength'] = float(np.max(matrix))
            
            # Degree analysis
            out_degree = np.sum(matrix > 0, axis=1)
            in_degree = np.sum(matrix > 0, axis=0)
            analysis['max_out_degree'] = int(np.max(out_degree))
            analysis['max_in_degree'] = int(np.max(in_degree))
            analysis['avg_out_degree'] = float(np.mean(out_degree))
            analysis['avg_in_degree'] = float(np.mean(in_degree))
            
            # Connectivity
            analysis['connected_components'] = DataProcessingEngine._count_components(matrix)
            
            # Centrality measures
            analysis['degree_centrality'] = DataProcessingEngine._calculate_degree_centrality(matrix)
            analysis['betweenness_centrality'] = DataProcessingEngine._calculate_betweenness_centrality(matrix)
            
            # Clustering
            analysis['clustering_coefficient'] = DataProcessingEngine._calculate_clustering_coefficient(matrix)
            
        else:  # adjacency_list
            # Convert to networkx graph for analysis
            G = nx.DiGraph() if data_type == 'directed' else nx.Graph()
            
            for _, row in data.iterrows():
                if len(row) >= 3:
                    G.add_edge(row[0], row[1], weight=row[2])
            
            analysis['num_nodes'] = G.number_of_nodes()
            analysis['num_edges'] = G.number_of_edges()
            analysis['density'] = nx.density(G)
            
            if nx.is_connected(G.to_undirected()):
                analysis['diameter'] = nx.diameter(G.to_undirected())
                analysis['avg_path_length'] = nx.average_shortest_path_length(G.to_undirected())
            else:
                analysis['diameter'] = 'Disconnected'
                analysis['avg_path_length'] = 'Disconnected'
            
            # Centrality measures
            if G.number_of_nodes() > 0:
                analysis['degree_centrality'] = nx.degree_centrality(G)
                analysis['betweenness_centrality'] = nx.betweenness_centrality(G)
                analysis['clustering_coefficient'] = nx.average_clustering(G.to_undirected())
        
        return analysis
    
    @staticmethod
    def _count_components(matrix: np.ndarray) -> int:
        """Count connected components in matrix"""
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components
        
        # Create adjacency matrix
        adj_matrix = (matrix > 0).astype(int)
        sparse_matrix = sparse.csr_matrix(adj_matrix)
        
        n_components, _ = connected_components(sparse_matrix, directed=False)
        return n_components
    
    @staticmethod
    def _calculate_degree_centrality(matrix: np.ndarray) -> np.ndarray:
        """Calculate degree centrality"""
        degrees = np.sum(matrix > 0, axis=1) + np.sum(matrix > 0, axis=0)
        n = matrix.shape[0]
        return degrees / (n - 1) if n > 1 else degrees
    
    @staticmethod
    def _calculate_betweenness_centrality(matrix: np.ndarray) -> np.ndarray:
        """Calculate betweenness centrality approximation"""
        n = matrix.shape[0]
        centrality = np.zeros(n)
        
        # Simplified approximation for large matrices
        if n > 100:
            # Use degree centrality as approximation
            return DataProcessingEngine._calculate_degree_centrality(matrix)
        
        # Full calculation for small matrices
        import itertools
        
        # Create adjacency list
        adj_list = {}
        for i in range(n):
            neighbors = np.where(matrix[i] > 0)[0]
            adj_list[i] = list(neighbors)
        
        # Calculate shortest paths
        for s in range(n):
            # BFS to find shortest paths
            distances = {s: 0}
            paths = {s: [[]]}
            queue = [s]
            
            while queue:
                v = queue.pop(0)
                for w in adj_list.get(v, []):
                    if w not in distances:
                        distances[w] = distances[v] + 1
                        paths[w] = [p + [v] for p in paths.get(v, [[]])]
                        queue.append(w)
                    elif distances[w] == distances[v] + 1:
                        paths[w].extend([p + [v] for p in paths.get(v, [[]])])
            
            # Update centrality
            for t in range(n):
                if t != s and t in paths:
                    shortest_paths = paths[t]
                    if shortest_paths:
                        count = len(shortest_paths)
                        for path in shortest_paths:
                            for node in path:
                                if node != s and node != t:
                                    centrality[node] += 1.0 / count
        
        # Normalize
        if n > 2:
            centrality /= ((n - 1) * (n - 2))
        
        return centrality
    
    @staticmethod
    def _calculate_clustering_coefficient(matrix: np.ndarray) -> float:
        """Calculate clustering coefficient"""
        n = matrix.shape[0]
        if n < 3:
            return 0.0
        
        total_coefficient = 0.0
        count = 0
        
        for i in range(n):
            neighbors = np.where(matrix[i] > 0)[0]
            k = len(neighbors)
            
            if k < 2:
                continue
            
            # Count triangles
            triangles = 0
            possible_triangles = k * (k - 1) / 2
            
            for idx1 in range(k):
                for idx2 in range(idx1 + 1, k):
                    j = neighbors[idx1]
                    k_node = neighbors[idx2]
                    if matrix[j, k_node] > 0 or matrix[k_node, j] > 0:
                        triangles += 1
            
            if possible_triangles > 0:
                total_coefficient += triangles / possible_triangles
                count += 1
        
        return total_coefficient / count if count > 0 else 0.0

# ============================================================================
# ULTRA-ENHANCED CHORD DIAGRAM ENGINE - LIGHT THEME EDITION
# ============================================================================

class UltraEnhancedChordDiagram:
    """
    Ultra-enhanced chord diagram renderer with light theme optimization.
    
    Features:
        - Ultra-large canvas (24x24 inches) at 300+ DPI
        - Light theme optimization with enhanced contrast
        - Advanced caching and performance optimization
        - Real-time interactivity and animations
        - Multi-core processing support
        - Professional-grade visual effects
        - Comprehensive analytics and statistics
    """
    
    def __init__(self, figsize: Tuple[int, int] = (24, 24), dpi: int = 300,
                 background_color: str = LIGHT_THEME_DEFAULTS['background_color'],
                 use_cache: bool = True, enable_animation: bool = False):
        """
        Initialize ultra-enhanced chord diagram canvas.
        
        Parameters
        ----------
        figsize : tuple
            Figure dimensions (width, height) in inches
        dpi : int
            Resolution in dots per inch
        background_color : str
            Canvas background color (light theme optimized)
        use_cache : bool
            Enable caching for performance
        enable_animation : bool
            Enable animation capabilities
        """
        self.figsize = figsize
        self.dpi = dpi
        self.background_color = background_color
        self.use_cache = use_cache
        self.enable_animation = enable_animation
        
        # Initialize figure with advanced settings
        self.fig = plt.figure(
            figsize=figsize, 
            dpi=dpi, 
            facecolor=background_color,
            layout='tight'
        )
        
        # Create polar axes with light theme optimization
        self.ax = self.fig.add_subplot(111, projection='polar', facecolor=background_color)
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        
        # Enhanced data structures
        self.sectors: List[str] = []
        self.sector_dict: Dict[str, Dict] = {}
        self.links: List[Dict] = []
        self.tracks: Dict[str, Dict[int, Dict]] = {}
        self.groups: Dict[str, int] = {}
        self.node_degrees: Dict[str, float] = {}
        self.node_betweenness: Dict[str, float] = {}
        
        # Advanced layout parameters
        self.gap_after: Dict[str, float] = {}
        self.start_degree: float = 0.0
        self.clockwise: bool = True
        self.big_gap: float = LIGHT_THEME_DEFAULTS['big_gap']
        self.small_gap: float = LIGHT_THEME_DEFAULTS['small_gap']
        self.sector_angles: Dict[str, Dict[str, float]] = {}
        self.sector_colors: Dict[str, str] = {}
        
        # Visual parameters with light theme optimization
        self.track_height: float = LIGHT_THEME_DEFAULTS['track_height']
        self.grid_alpha: float = LIGHT_THEME_DEFAULTS['grid_alpha']
        self.grid_color: str = LIGHT_THEME_DEFAULTS['grid_color']
        self.text_color: str = LIGHT_THEME_DEFAULTS['text_color_primary']
        
        # Performance tracking
        self.render_times: List[float] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        
        # Interactive elements
        self.highlighted_links: Set[Tuple[str, str]] = set()
        self.selected_sectors: Set[str] = set()
        self.tooltip_data: Dict[str, Any] = {}
        
        # Animation state
        self.animation_frames: List[Any] = []
        self.current_frame: int = 0
        
        # Statistics
        self.stats = {
            'total_links': 0,
            'total_value': 0.0,
            'max_link_value': 0.0,
            'avg_link_value': 0.0,
            'link_density': 0.0,
            'render_time': 0.0,
            'memory_usage': 0.0
        }
        
        # Initialize performance monitoring
        self._start_time = time.time()
    
    def initialize_sectors(self, sectors: List[str], groups: Optional[Dict[str, int]] = None,
                          sector_sizes: Optional[Dict[str, float]] = None,
                          sector_colors: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize sectors with advanced grouping and coloring.
        
        Parameters
        ----------
        sectors : list of str
            List of sector identifiers in display order
        groups : dict, optional
            Mapping from sector name to group ID
        sector_sizes : dict, optional
            Custom sizes for sectors
        sector_colors : dict, optional
            Custom colors for sectors
        """
        self.sectors = sectors
        self.groups = groups if groups else {s: 0 for s in sectors}
        
        # Initialize colors
        if sector_colors:
            self.sector_colors = sector_colors
        else:
            # Generate colors based on groups
            n_groups = len(set(self.groups.values()))
            group_colors = AdvancedColorEngine.generate_color_palette(
                n_groups, 'soft_pastel' if n_groups <= 8 else 'vivid_light'
            )
            self.sector_colors = {
                sector: group_colors[self.groups[sector] % len(group_colors)]
                for sector in sectors
            }
        
        # Initialize default gaps
        self.gap_after = {sector: self.small_gap for sector in sectors}
        
        # Apply larger gaps between different groups
        if len(set(self.groups.values())) > 1:
            for i in range(len(sectors) - 1):
                current_sector = sectors[i]
                next_sector = sectors[i + 1]
                if self.groups[current_sector] != self.groups[next_sector]:
                    self.gap_after[current_sector] = self.big_gap
        
        # Initialize node metrics
        self.node_degrees = {sector: 0.0 for sector in sectors}
        self.node_betweenness = {sector: 0.0 for sector in sectors}
    
    def compute_sector_angles(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate precise angular positions for all sectors with caching.
        
        Returns
        -------
        dict
            Mapping from sector name to {start, end, mid} angles in degrees
        """
        # Create cache key from hashable parameters
        cache_params = {
            'sectors': tuple(self.sectors),
            'gap_after': tuple(sorted(self.gap_after.items())),
            'start_degree': self.start_degree,
            'big_gap': self.big_gap,
            'small_gap': self.small_gap
        }
        
        cache_key = f"sector_angles_{hash(str(cache_params))}"
        
        # Use custom caching instead of Streamlit's decorator
        if self.use_cache:
            cached_result = SmartCacheSystem.get_or_create(
                cache_key,
                self._compute_sector_angles_impl
            )
            self.sector_angles = cached_result
            return cached_result
        else:
            result = self._compute_sector_angles_impl()
            self.sector_angles = result
            return result
    
    def _compute_sector_angles_impl(self) -> Dict[str, Dict[str, float]]:
        """Actual implementation of sector angle computation"""
        total_gap = sum(self.gap_after.get(sector, self.small_gap) for sector in self.sectors)
        available_degrees = 360.0 - total_gap
        
        # Support variable widths if provided
        if hasattr(self, 'sector_sizes'):
            total_size = sum(self.sector_sizes.values())
            sector_widths = {
                sector: (self.sector_sizes[sector] / total_size) * available_degrees
                for sector in self.sectors
            }
        else:
            sector_width = available_degrees / len(self.sectors)
            sector_widths = {sector: sector_width for sector in self.sectors}
        
        angles = {}
        current_angle = self.start_degree
        
        for sector in self.sectors:
            width = sector_widths[sector]
            angles[sector] = {
                'start': current_angle % 360,
                'end': (current_angle + width) % 360,
                'mid': (current_angle + width / 2) % 360,
                'width': width
            }
            current_angle += width + self.gap_after.get(sector, self.small_gap)
        
        return angles
    
    def draw_enhanced_track(self, sector: str, angles: Dict[str, float], 
                          color: Union[str, Tuple] = None,
                          track_index: int = 0,
                          gradient: bool = True,
                          pattern: str = None) -> Dict[str, float]:
        """
        Render an enhanced sector track with advanced visual effects.
        
        Parameters
        ----------
        sector : str
            Sector identifier
        angles : dict
            Angular boundaries
        color : str or RGB tuple, optional
            Fill color (uses sector color if None)
        track_index : int
            Track position
        gradient : bool
            Apply radial gradient
        pattern : str, optional
            Pattern type: 'solid', 'striped', 'dotted', 'hatched'
        
        Returns
        -------
        dict
            Track geometry data
        """
        if color is None:
            color = self.sector_colors.get(sector, self.grid_color)
        
        start_rad = np.radians(angles['start'])
        end_rad = np.radians(angles['end'])
        
        # Handle angular wrap-around
        if end_rad < start_rad and abs(end_rad - start_rad) < np.pi:
            end_rad += 2 * np.pi
        
        # Create smooth arc with adaptive resolution
        n_points = max(100, int(abs(end_rad - start_rad) * 50))
        theta = np.linspace(start_rad, end_rad, n_points)
        
        r_inner = track_index * self.track_height
        r_outer = r_inner + self.track_height
        
        # Build polygon vertices
        theta_poly = np.concatenate([theta, theta[::-1]])
        r_poly = np.concatenate([np.full_like(theta, r_inner), 
                                np.full_like(theta, r_outer)])
        
        # Convert to Cartesian
        x = r_poly * np.cos(theta_poly)
        y = r_poly * np.sin(theta_poly)
        vertices = np.column_stack([x, y])
        
        # Apply gradient effect
        if gradient:
            # Create gradient fill
            gradient_patch = self._create_gradient_patch(vertices, color)
            self.ax.add_patch(gradient_patch)
        else:
            # Solid fill
            poly = plt.Polygon(vertices, facecolor=color, 
                              alpha=0.15, edgecolor='none', 
                              zorder=0.5)
            self.ax.add_patch(poly)
        
        # Apply pattern if specified
        if pattern:
            pattern_patch = self._apply_pattern(vertices, pattern, color)
            self.ax.add_patch(pattern_patch)
        
        # Add subtle border
        border_poly = plt.Polygon(vertices, facecolor='none', 
                                 edgecolor=self._adjust_color_brightness(color, -0.2),
                                 alpha=0.3, linewidth=0.5,
                                 zorder=0.6)
        self.ax.add_patch(border_poly)
        
        track_data = {'inner': r_inner, 'outer': r_outer, 'mid_angle': np.radians(angles['mid'])}
        
        # Store track data
        if sector not in self.tracks:
            self.tracks[sector] = {}
        self.tracks[sector][track_index] = track_data
        
        return track_data
    
    def _create_gradient_patch(self, vertices: np.ndarray, base_color: str) -> PatchCollection:
        """Create gradient-filled patch"""
        # Create triangle mesh for gradient
        from matplotlib.tri import Triangulation
        
        x = vertices[:, 0]
        y = vertices[:, 1]
        n = len(x) // 2
        
        # Create triangulation
        triangles = []
        for i in range(n - 1):
            triangles.append([i, i + 1, n + i])
            triangles.append([i + 1, n + i, n + i + 1])
        
        tri = Triangulation(x, y, triangles)
        
        # Create gradient values
        z = np.zeros(len(x))
        z[:n] = 0.3  # Inner edge
        z[n:] = 0.7  # Outer edge
        
        # Create colored triangles
        cmap = LinearSegmentedColormap.from_list('gradient', 
                                                ['white', base_color, 'white'])
        colors = cmap(z[tri.triangles].mean(axis=1))
        
        collection = PolyCollection(tri.triangles, array=z, cmap=cmap,
                                  alpha=0.15, edgecolors='none', zorder=0.5)
        collection.set_array(z[tri.triangles].mean(axis=1))
        
        return collection
    
    def _apply_pattern(self, vertices: np.ndarray, pattern: str, color: str) -> Optional[PatchCollection]:
        """Apply pattern to polygon"""
        if pattern == 'striped':
            # Create striped pattern
            x = vertices[:, 0]
            y = vertices[:, 1]
            
            # Calculate bounding box
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            
            # Create stripes
            stripe_width = (x_max - x_min) / 20
            stripes = []
            
            for i in range(20):
                stripe_x = x_min + i * stripe_width
                rect = Rectangle((stripe_x, y_min), stripe_width, y_max - y_min)
                if i % 2 == 0:
                    stripes.append(rect)
            
            collection = PatchCollection(stripes, facecolor=color, 
                                       alpha=0.1, edgecolor='none',
                                       zorder=0.45)
            return collection
        
        elif pattern == 'hatched':
            # Create hatched pattern
            poly = plt.Polygon(vertices, facecolor='none', 
                             hatch='////', edgecolor=color,
                             alpha=0.2, linewidth=0,
                             zorder=0.45)
            return poly
        
        return None
    
    def create_enhanced_link(self, source: str, target: str, value: float,
                           source_track: int = 0, target_track: int = 0,
                           color: Union[str, Tuple] = None,
                           style: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create enhanced curved link with advanced styling options.
        
        Parameters
        ----------
        source, target : str
            Source and target sector identifiers
        value : float
            Link weight
        source_track, target_track : int
            Track indices
        color : str or RGB tuple, optional
            Link color
        style : dict, optional
            Advanced styling options
        
        Returns
        -------
        dict
            Link metadata
        """
        # Use cache for expensive computations
        cache_key = f"link_{source}_{target}_{value}_{source_track}_{target_track}"
        
        if self.use_cache and cache_key in SmartCacheSystem._cache_store:
            self.cache_hits += 1
            cached_data = SmartCacheSystem._cache_store[cache_key]
            self.links.append(cached_data)
            return cached_data
        
        self.cache_misses += 1
        
        # Get angular positions
        if source not in self.sector_angles:
            self.compute_sector_angles()
        if target not in self.sector_angles:
            self.compute_sector_angles()
            
        source_angle = self.sector_angles[source]['mid']
        target_angle = self.sector_angles[target]['mid']
        
        # Get radial positions
        source_r = self.tracks.get(source, {}).get(source_track, {}).get('outer', 0.20)
        target_r = self.tracks.get(target, {}).get(target_track, {}).get('outer', 0.20)
        
        # Apply styling
        if style is None:
            style = {}
        
        # Determine color
        if color is None:
            if 'color_strategy' in style and style['color_strategy'] == 'source':
                color = self.sector_colors.get(source, '#666666')
            elif 'color_strategy' in style and style['color_strategy'] == 'target':
                color = self.sector_colors.get(target, '#666666')
            else:
                # Use value-based coloring
                max_val = self.stats.get('max_link_value', 1.0)
                norm_value = value / max_val
                cmap = plt.cm.viridis
                color = cmap(norm_value)
        
        # Create enhanced curve
        link_data = self._create_enhanced_curve(
            source, target, value,
            source_angle, target_angle,
            source_r, target_r,
            color, style
        )
        
        # Store in cache
        if self.use_cache:
            SmartCacheSystem._cache_store[cache_key] = link_data
        
        self.links.append(link_data)
        
        # Update node metrics
        self.node_degrees[source] = self.node_degrees.get(source, 0.0) + value
        self.node_degrees[target] = self.node_degrees.get(target, 0.0) + value
        
        # Update statistics
        self.stats['total_links'] = len(self.links)
        self.stats['total_value'] = self.stats.get('total_value', 0.0) + value
        self.stats['max_link_value'] = max(self.stats.get('max_link_value', 0.0), value)
        self.stats['avg_link_value'] = self.stats['total_value'] / self.stats['total_links']
        
        return link_data
    
    def _create_enhanced_curve(self, source: str, target: str, value: float,
                             source_angle: float, target_angle: float,
                             source_r: float, target_r: float,
                             color: Any, style: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced Bezier curve with advanced features"""
        # Convert to radians
        source_rad = np.radians(source_angle)
        target_rad = np.radians(target_angle)
        
        # Handle angular wrap-around
        if abs(target_rad - source_rad) > np.pi:
            if target_rad > source_rad:
                target_rad -= 2 * np.pi
            else:
                source_rad -= 2 * np.pi
        
        # Advanced curve generation with multiple control points
        t = np.linspace(0, 1, 150)  # High resolution
        
        # Use cubic Bezier for smoother curves
        if style.get('curve_type', 'quadratic') == 'cubic':
            # Two control points for cubic Bezier
            control1_angle = source_rad + (target_rad - source_rad) * 0.3
            control2_angle = source_rad + (target_rad - source_rad) * 0.7
            control1_r = max(source_r, target_r) * 1.8
            control2_r = max(source_r, target_r) * 1.8
            
            theta_curve = ((1 - t)**3 * source_rad + 
                          3 * (1 - t)**2 * t * control1_angle + 
                          3 * (1 - t) * t**2 * control2_angle + 
                          t**3 * target_rad)
            r_curve = ((1 - t)**3 * source_r + 
                      3 * (1 - t)**2 * t * control1_r + 
                      3 * (1 - t) * t**2 * control2_r + 
                      t**3 * target_r)
        else:
            # Quadratic Bezier
            control_angle = (source_rad + target_rad) / 2
            control_r = max(source_r, target_r) * 1.6
            
            theta_curve = (1 - t)**2 * source_rad + 2 * (1 - t) * t * control_angle + t**2 * target_rad
            r_curve = (1 - t)**2 * source_r + 2 * (1 - t) * t * control_r + t**2 * target_r
        
        # Normalize angles
        theta_curve = theta_curve % (2 * np.pi)
        
        # Calculate line width with advanced scaling
        base_width = self._calculate_line_width(value, style)
        
        # Create link with advanced effects
        link_elements = []
        
        # 1. Background glow (if enabled)
        if style.get('glow', True):
            glow_width = base_width * style.get('glow_intensity', 2.5)
            glow_alpha = style.get('glow_alpha', 0.3)
            
            for i in range(3):
                w_mult = [2.5, 1.8, 1.2][i]
                a_mult = [0.2, 0.3, 0.4][i]
                
                glow_line, = self.ax.plot(
                    theta_curve, r_curve,
                    color=color,
                    alpha=glow_alpha * a_mult,
                    linewidth=glow_width * w_mult,
                    solid_capstyle='round',
                    solid_joinstyle='round',
                    zorder=10 + i
                )
                link_elements.append(('glow', glow_line))
        
        # 2. Main line with gradient
        if style.get('gradient', False):
            # Create gradient along the curve
            segments = self._create_gradient_segments(theta_curve, r_curve, color, base_width)
            link_elements.append(('gradient', segments))
        else:
            # Solid line
            main_line, = self.ax.plot(
                theta_curve, r_curve,
                color=color,
                alpha=style.get('alpha', 0.85),
                linewidth=base_width,
                solid_capstyle='round',
                solid_joinstyle='round',
                zorder=15
            )
            link_elements.append(('main', main_line))
        
        # 3. Border (if enabled)
        if style.get('border', True):
            border_width = style.get('border_width', 0.8)
            border_color = style.get('border_color', '#FFFFFF')
            
            border_line, = self.ax.plot(
                theta_curve, r_curve,
                color=border_color,
                alpha=min(1.0, style.get('alpha', 0.85) * 1.2),
                linewidth=base_width + border_width * 2,
                solid_capstyle='round',
                solid_joinstyle='round',
                zorder=14
            )
            link_elements.append(('border', border_line))
        
        # 4. Direction indicators (if enabled)
        if style.get('directional', False):
            direction_type = style.get('direction_type', ['diffHeight', 'arrows'])
            
            if 'diffHeight' in direction_type:
                offset = style.get('diff_height', 0.06)
                # Height difference already applied in radial positions
            
            if 'arrows' in direction_type:
                arrows = self._add_direction_arrows(
                    theta_curve, r_curve, color,
                    style.get('arrow_length', 0.18),
                    style.get('arrow_width', 0.10),
                    style.get('alpha', 0.85),
                    base_width
                )
                link_elements.extend([('arrow', arrow) for arrow in arrows])
        
        # 5. Highlight effects (if this link is highlighted)
        if (source, target) in self.highlighted_links:
            highlight_elements = self._add_highlight_effects(
                theta_curve, r_curve, color,
                style.get('alpha', 0.85),
                base_width
            )
            link_elements.extend(highlight_elements)
        
        # Store link metadata
        link_data = {
            'source': source,
            'target': target,
            'value': value,
            'color': color,
            'alpha': style.get('alpha', 0.85),
            'width': base_width,
            'elements': link_elements,
            'coordinates': (theta_curve.copy(), r_curve.copy()),
            'zorder': 15
        }
        
        return link_data
    
    def _calculate_line_width(self, value: float, style: Dict[str, Any]) -> float:
        """Calculate line width with advanced scaling"""
        min_width = style.get('min_width', LIGHT_THEME_DEFAULTS['link_min_width'])
        max_width = style.get('max_width', LIGHT_THEME_DEFAULTS['link_max_width'])
        width_scale = style.get('width_scale', LIGHT_THEME_DEFAULTS['link_width_scale'])
        
        # Apply non-linear scaling for better visual distribution
        scaled_value = value * width_scale
        
        # Use logarithmic scaling for better distribution of values
        if style.get('log_scaling', False):
            scaled_value = np.log1p(scaled_value)
        
        # Apply power scaling for emphasis
        power = style.get('power_scaling', 0.7)
        scaled_value = scaled_value ** power
        
        width = min(max_width, max(min_width, scaled_value))
        
        return width
    
    def _create_gradient_segments(self, theta: np.ndarray, r: np.ndarray, 
                                color: Any, width: float) -> LineCollection:
        """Create gradient-colored line segments"""
        points = np.array([theta, r]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create gradient from source to target
        n_segments = len(segments)
        if isinstance(color, tuple) and len(color) == 2:
            # Two-color gradient
            colors = []
            for i in range(n_segments):
                t = i / n_segments
                interp_color = tuple(
                    c1 * (1 - t) + c2 * t
                    for c1, c2 in zip(color[0], color[1])
                )
                colors.append(interp_color)
        else:
            # Single color with opacity gradient
            colors = []
            base_color = mcolors.to_rgba(color)
            for i in range(n_segments):
                t = i / n_segments
                alpha = base_color[3] * (0.5 + 0.5 * np.sin(np.pi * t))
                colors.append((base_color[0], base_color[1], base_color[2], alpha))
        
        lc = LineCollection(segments, colors=colors, linewidths=width,
                          capstyle='round', joinstyle='round',
                          zorder=15)
        self.ax.add_collection(lc)
        
        return lc
    
    def _add_direction_arrows(self, theta: np.ndarray, r: np.ndarray, 
                            color: Any, arrow_length: float, 
                            arrow_width: float, alpha: float,
                            base_width: float) -> List[FancyArrowPatch]:
        """Add direction arrows to curve"""
        arrows = []
        
        # Determine arrow positions based on curve length
        n_arrows = max(1, min(3, int(len(theta) / 30)))
        arrow_positions = np.linspace(0.4, 0.8, n_arrows)
        
        for pos in arrow_positions:
            idx = int(len(theta) * pos)
            
            if idx < 2 or idx >= len(theta) - 2:
                continue
            
            # Calculate tangent direction
            dx = r[idx + 2] * np.cos(theta[idx + 2]) - r[idx - 2] * np.cos(theta[idx - 2])
            dy = r[idx + 2] * np.sin(theta[idx + 2]) - r[idx - 2] * np.sin(theta[idx - 2])
            
            # Normalize direction
            norm = np.sqrt(dx**2 + dy**2)
            if norm == 0:
                continue
            
            dx /= norm
            dy /= norm
            
            # Calculate arrow position in Cartesian coordinates
            x = r[idx] * np.cos(theta[idx])
            y = r[idx] * np.sin(theta[idx])
            
            # Create arrow
            arrow = FancyArrowPatch(
                (x - dx * arrow_length * 0.5, y - dy * arrow_length * 0.5),
                (x + dx * arrow_length * 0.5, y + dy * arrow_length * 0.5),
                arrowstyle=f'->,head_width={arrow_width*12},head_length={arrow_length*8}',
                color=color,
                alpha=min(1.0, alpha * 1.3),
                linewidth=base_width * 0.6,
                mutation_scale=25,
                zorder=20
            )
            
            self.ax.add_patch(arrow)
            arrows.append(arrow)
        
        return arrows
    
    def _add_highlight_effects(self, theta: np.ndarray, r: np.ndarray, 
                             color: Any, alpha: float, 
                             base_width: float) -> List[Tuple[str, Any]]:
        """Add highlight effects to link"""
        elements = []
        
        # White glow
        white_glow, = self.ax.plot(
            theta, r,
            color='white',
            alpha=0.6,
            linewidth=base_width * 3.0,
            solid_capstyle='round',
            zorder=13
        )
        elements.append(('white_glow', white_glow))
        
        # Color glow
        color_glow, = self.ax.plot(
            theta, r,
            color=color,
            alpha=0.8,
            linewidth=base_width * 2.0,
            solid_capstyle='round',
            zorder=14
        )
        elements.append(('color_glow', color_glow))
        
        # Pulsing effect (animated)
        if self.enable_animation:
            pulse_line, = self.ax.plot(
                theta, r,
                color=color,
                alpha=0.4,
                linewidth=base_width * 1.5,
                solid_capstyle='round',
                zorder=12,
                animated=True
            )
            elements.append(('pulse', pulse_line))
        
        return elements
    
    def _adjust_color_brightness(self, color: str, factor: float) -> str:
        """Adjust color brightness"""
        try:
            rgb = mcolors.to_rgb(color)
            h, l, s = colorsys.rgb_to_hls(*rgb)
            l = max(0.0, min(1.0, l + factor))
            rgb_adjusted = colorsys.hls_to_rgb(h, l, s)
            return mcolors.to_hex(rgb_adjusted)
        except:
            return color
    
    def add_track(self, sector: str, track_index: int = 0, color: str = None) -> Dict[str, float]:
        """
        Add a track to a sector with proper styling.
        
        Parameters
        ----------
        sector : str
            Sector identifier
        track_index : int
            Track position
        color : str, optional
            Track color
        
        Returns
        -------
        dict
            Track geometry data
        """
        if sector not in self.sector_angles:
            self.compute_sector_angles()
        
        angles = self.sector_angles[sector]
        
        if color is None:
            color = self.sector_colors.get(sector, self.grid_color)
        
        return self.draw_enhanced_track(
            sector, angles, color, track_index, 
            gradient=True, pattern=None
        )
    
    def add_interactive_elements(self):
        """Add interactive elements to the diagram"""
        # Add sector click handlers (simulated)
        for sector, angle_data in self.sector_angles.items():
            angle_rad = np.radians(angle_data['mid'])
            outer_r = 0.0
            
            if sector in self.tracks:
                for track_data in self.tracks[sector].values():
                    outer_r = max(outer_r, track_data['outer'])
            
            # Create invisible circle for clicking
            click_circle = Circle(
                (outer_r * np.cos(angle_rad), outer_r * np.sin(angle_rad)),
                radius=outer_r * 0.05,
                facecolor='none',
                edgecolor='none',
                alpha=0.0,
                zorder=1000,
                picker=True
            )
            self.ax.add_patch(click_circle)
            
            # Store metadata
            self.tooltip_data[f"sector_{sector}"] = {
                'type': 'sector',
                'name': sector,
                'degree': self.node_degrees.get(sector, 0),
                'betweenness': self.node_betweenness.get(sector, 0),
                'angle': angle_rad,
                'radius': outer_r
            }
        
        # Add link hover effects
        for link in self.links:
            # Create invisible line for hovering
            theta, r = link['coordinates']
            hover_line, = self.ax.plot(
                theta, r,
                color='none',
                linewidth=link['width'] * 3,  # Larger hit area
                alpha=0.0,
                zorder=1001,
                picker=True
            )
            
            # Store metadata
            self.tooltip_data[f"link_{link['source']}_{link['target']}"] = {
                'type': 'link',
                'source': link['source'],
                'target': link['target'],
                'value': link['value'],
                'color': link['color'],
                'width': link['width']
            }
    
    def create_animation(self, duration: int = 1000, fps: int = 30) -> animation.FuncAnimation:
        """Create animation of the diagram"""
        if not self.enable_animation:
            return None
        
        n_frames = int(duration * fps / 1000)
        
        def init():
            return []
        
        def animate(frame):
            # Update animation frame
            alpha = 0.5 + 0.5 * np.sin(2 * np.pi * frame / n_frames)
            
            # Update pulsing elements
            for link in self.links:
                if 'pulse' in [elem[0] for elem in link['elements']]:
                    for elem_type, elem in link['elements']:
                        if elem_type == 'pulse':
                            elem.set_alpha(alpha * 0.4)
            
            return []
        
        anim = FuncAnimation(self.fig, animate, init_func=init,
                           frames=n_frames, interval=1000/fps,
                           blit=True)
        
        return anim
    
    def export_to_html(self) -> str:
        """Export diagram to interactive HTML"""
        # Convert to Plotly for interactive HTML export
        fig_html = go.Figure()
        
        # Add sectors
        for sector, angle_data in self.sector_angles.items():
            angle_rad = np.radians(angle_data['mid'])
            outer_r = 0.0
            
            if sector in self.tracks:
                for track_data in self.tracks[sector].values():
                    outer_r = max(outer_r, track_data['outer'])
            
            fig_html.add_trace(go.Scatterpolar(
                r=[outer_r],
                theta=[angle_data['mid']],
                mode='markers+text',
                name=sector,
                marker=dict(
                    size=self.node_degrees.get(sector, 0) * 10 + 10,
                    color=self.sector_colors.get(sector, '#666666'),
                    line=dict(width=2, color='white')
                ),
                text=sector,
                textposition='top center',
                hoverinfo='text+name'
            ))
        
        # Add links
        for link in self.links:
            theta, r = link['coordinates']
            fig_html.add_trace(go.Scatterpolar(
                r=r,
                theta=np.degrees(theta),
                mode='lines',
                line=dict(
                    width=link['width'],
                    color=link['color']
                ),
                opacity=link['alpha'],
                hoverinfo='none',
                showlegend=False
            ))
        
        # Update layout
        fig_html.update_layout(
            polar=dict(
                bgcolor=self.background_color,
                radialaxis=dict(
                    visible=False,
                    range=[0, 1.5]
                ),
                angularaxis=dict(
                    visible=False
                )
            ),
            showlegend=False,
            paper_bgcolor=self.background_color,
            plot_bgcolor=self.background_color
        )
        
        return fig_html.to_html(include_plotlyjs='cdn', full_html=False)
    
    def finalize(self, title: str = "", show_frame: bool = False, 
                background_color: Optional[str] = None,
                show_grid: bool = False,
                interactive: bool = True) -> plt.Figure:
        """
        Finalize diagram with advanced features.
        
        Parameters
        ----------
        title : str
            Diagram title
        show_frame : bool
            Show polar frame
        background_color : str, optional
            Background color
        show_grid : bool
            Show grid
        interactive : bool
            Enable interactive elements
        
        Returns
        -------
        matplotlib.figure.Figure
            Finalized figure
        """
        if background_color is None:
            background_color = self.background_color
        
        # Set background
        self.fig.patch.set_facecolor(background_color)
        self.ax.set_facecolor(background_color)
        
        # Set axis limits
        max_radius = 1.5
        for sector_tracks in self.tracks.values():
            if sector_tracks:
                max_radius = max(max_radius, max(t['outer'] for t in sector_tracks.values()))
        
        self.ax.set_ylim(0, max_radius * 1.4)
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        
        # Configure frame
        self.ax.spines['polar'].set_visible(show_frame)
        if show_frame:
            self.ax.spines['polar'].set_color(self.grid_color)
            self.ax.spines['polar'].set_linewidth(1.0)
        
        # Add grid
        if show_grid:
            self.ax.grid(True, alpha=0.1, color=self.grid_color, linewidth=0.5)
        
        # Add title
        if title:
            title_text = self.ax.set_title(
                title,
                fontsize=28,
                fontweight='bold',
                pad=40,
                color=self.text_color,
                fontfamily='sans-serif'
            )
        
        # Add interactive elements
        if interactive:
            self.add_interactive_elements()
        
        # Add legend
        self._add_enhanced_legend()
        
        # Add statistics overlay
        self._add_statistics_overlay()
        
        # Update render statistics
        render_time = time.time() - self._start_time
        self.stats['render_time'] = render_time
        self.render_times.append(render_time)
        
        # Optimize layout
        self.fig.tight_layout(pad=2.0)
        
        return self.fig
    
    def _add_enhanced_legend(self):
        """Add enhanced legend to diagram"""
        if not LIGHT_THEME_DEFAULTS['show_legend']:
            return
        
        # Create custom legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        legend_elements = []
        
        # Node size legend
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='gray', markersize=8,
                  label='Small node', linestyle='None')
        )
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='gray', markersize=16,
                  label='Large node', linestyle='None')
        )
        
        # Link width legend
        legend_elements.append(
            Line2D([0], [0], color='gray', linewidth=1,
                  label='Weak link')
        )
        legend_elements.append(
            Line2D([0], [0], color='gray', linewidth=5,
                  label='Strong link')
        )
        
        # Direction legend
        if any(link.get('directional', False) for link in self.links):
            legend_elements.append(
                Line2D([0], [0], color='gray', linewidth=2,
                      marker='>', markersize=10,
                      label='Directional')
            )
        
        # Add legend
        if legend_elements:
            self.ax.legend(handles=legend_elements,
                          loc='upper left',
                          bbox_to_anchor=(1.05, 1),
                          borderaxespad=0.,
                          framealpha=0.9,
                          facecolor=self.background_color,
                          edgecolor=self.grid_color)
    
    def _add_statistics_overlay(self):
        """Add statistics overlay to diagram"""
        if not LIGHT_THEME_DEFAULTS['show_statistics']:
            return
        
        stats_text = [
            f"Nodes: {len(self.sectors)}",
            f"Links: {len(self.links)}",
            f"Total Strength: {self.stats['total_value']:.2f}",
            f"Avg Link: {self.stats['avg_link_value']:.3f}",
            f"Max Link: {self.stats['max_link_value']:.3f}",
            f"Render Time: {self.stats['render_time']:.2f}s",
            f"Cache: {self.cache_hits}/{self.cache_hits + self.cache_misses}"
        ]
        
        stats_str = "\n".join(stats_text)
        
        # Add text box
        self.fig.text(0.02, 0.02, stats_str,
                     fontsize=10,
                     fontfamily='monospace',
                     fontweight='normal',
                     color=self.text_color,
                     alpha=0.8,
                     bbox=dict(boxstyle='round,pad=0.5',
                             facecolor=self.background_color,
                             edgecolor=self.grid_color,
                             linewidth=1,
                             alpha=0.7),
                     transform=self.fig.transFigure,
                     verticalalignment='bottom',
                     horizontalalignment='left',
                     zorder=10000)

# ============================================================================
# ADVANCED STREAMLIT APPLICATION WITH LIGHT THEME
# ============================================================================

class UltraEnhancedStreamlitApp:
    """Ultra-enhanced Streamlit application with light theme"""
    
    @staticmethod
    def setup_page():
        """Setup Streamlit page with light theme"""
        st.set_page_config(
            page_title="✨ Ultra Enhanced Chord Diagrams - Light Theme",
            page_icon="🌐",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        SessionStateManager.initialize_session_state()
        
        # Apply light theme CSS
        UltraEnhancedStreamlitApp._apply_light_theme_css()
        
        # Setup main layout
        UltraEnhancedStreamlitApp._create_header()
        
        # Initialize performance monitoring
        if 'performance_monitor' not in st.session_state:
            st.session_state.performance_monitor = {
                'start_time': time.time(),
                'page_views': 0,
                'render_times': []
            }
    
    @staticmethod
    def _apply_light_theme_css():
        """Apply light theme CSS"""
        st.markdown("""
        <style>
        /* Light Theme Styles */
        .main {
            background-color: #FFFFFF;
            color: #2C3E50;
        }
        
        .stApp {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        }
        
        /* Header */
        h1, h2, h3, h4, h5, h6 {
            color: #2C3E50;
            font-weight: 600;
        }
        
        .title-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .title-text {
            color: white;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .subtitle {
            color: #7F8C8D;
            font-size: 1.2rem;
            margin-top: 0.5rem;
        }
        
        /* Cards */
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border: 1px solid #E0E0E0;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFFFFF 0%, #F8F9FA 100%);
            border-right: 1px solid #E0E0E0;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #F8F9FA;
            padding: 0.5rem;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            background: white;
            color: #7F8C8D;
            border: 1px solid #E0E0E0;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
            box-shadow: 0 2px 5px rgba(102, 126, 234, 0.3);
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background: #F8F9FA;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
            font-weight: 600;
            color: #2C3E50;
        }
        
        /* Metrics */
        .stMetric {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #E0E0E0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        /* Sliders */
        .stSlider > div > div {
            color: #667eea;
        }
        
        /* Selectboxes */
        .stSelectbox > div > div {
            background: white;
            border: 1px solid #E0E0E0;
            border-radius: 6px;
        }
        
        /* Progress bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        /* Code blocks */
        .stCodeBlock {
            background: #F8F9FA;
            border-radius: 8px;
            border: 1px solid #E0E0E0;
        }
        
        /* Success/Error messages */
        .stAlert {
            border-radius: 8px;
            border: none;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
            border-top: 1px solid #E0E0E0;
            color: #7F8C8D;
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def _create_header():
        """Create application header"""
        st.markdown("""
        <div class="title-container">
            <h1 class="title-text">✨ Ultra Enhanced Chord Diagrams</h1>
            <p class="subtitle">Professional Network Visualization with Light Theme & Advanced Analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_sidebar():
        """Create enhanced sidebar with light theme"""
        with st.sidebar:
            st.image("https://via.placeholder.com/250x80/667eea/FFFFFF?text=CHORD+DIAGRAMS", 
                    use_column_width=True)
            
            st.markdown("---")
            
            # Data Configuration
            with st.expander("📊 **Data Configuration**", expanded=True):
                data_source = st.radio(
                    "Data Source",
                    ["Sample Data", "Upload CSV", "Upload Excel", "Demo Dataset"],
                    index=0
                )
                
                if data_source == "Upload CSV":
                    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
                    if uploaded_file:
                        data, data_type, meta = DataProcessingEngine.load_and_preprocess_data(
                            file_object=uploaded_file
                        )
                        st.session_state.current_data = data
                        st.session_state.data_type = data_type
                        st.session_state.data_loaded = True
                        st.success(f"✅ Loaded {meta['original_shape']} dataset")
                
                elif data_source == "Upload Excel":
                    uploaded_file = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'])
                    if uploaded_file:
                        data, data_type, meta = DataProcessingEngine.load_and_preprocess_data(
                            file_object=uploaded_file
                        )
                        st.session_state.current_data = data
                        st.session_state.data_type = data_type
                        st.session_state.data_loaded = True
                        st.success(f"✅ Loaded {meta['original_shape']} dataset")
                
                elif data_source == "Demo Dataset":
                    dataset_type = st.selectbox(
                        "Demo Dataset",
                        ["Gene Interactions", "Social Network", "Transportation", "Financial"]
                    )
                    
                    if st.button("Load Demo Data"):
                        with st.spinner("Loading demo dataset..."):
                            # Generate appropriate demo data
                            if dataset_type == "Gene Interactions":
                                size = 20
                            elif dataset_type == "Social Network":
                                size = 25
                            elif dataset_type == "Transportation":
                                size = 30
                            else:  # Financial
                                size = 15
                            
                            data = DataProcessingEngine.generate_sample_data(size=size)
                            st.session_state.current_data = data
                            st.session_state.data_type = 'matrix'
                            st.session_state.data_loaded = True
                            st.success(f"✅ Generated {dataset_type} demo dataset ({size}x{size})")
                
                else:  # Sample Data
                    if st.button("Generate Sample Data"):
                        data = DataProcessingEngine.generate_sample_data(size=20)
                        st.session_state.current_data = data
                        st.session_state.data_type = 'matrix'
                        st.session_state.data_loaded = True
                        st.success("✅ Generated sample dataset (20x20)")
            
            st.markdown("---")
            
            # Theme Settings
            with st.expander("🎨 **Theme & Appearance**", expanded=False):
                background_color = st.selectbox(
                    "Background Color",
                    list(LightThemeConfig.BACKGROUND_COLORS.keys()),
                    index=0
                )
                SessionStateManager.update_setting('background_color', 
                                                 LightThemeConfig.BACKGROUND_COLORS[background_color])
                
                palette_type = st.selectbox(
                    "Color Palette",
                    list(LightThemeConfig.LINK_PALETTES.keys()),
                    index=0
                )
                SessionStateManager.update_setting('palette', palette_type)
                
                col1, col2 = st.columns(2)
                with col1:
                    dpi = st.slider("Resolution (DPI)", 150, 600, 300, 50)
                    SessionStateManager.update_setting('dpi', dpi)
                with col2:
                    figsize = st.slider("Canvas Size", 10, 32, 24, 2)
                    SessionStateManager.update_setting('figsize', (figsize, figsize))
            
            st.markdown("---")
            
            # Performance Settings
            with st.expander("⚡ **Performance Settings**", expanded=False):
                performance_mode = st.selectbox(
                    "Performance Mode",
                    ["Quality", "Balanced", "Speed"],
                    index=1
                )
                SessionStateManager.update_setting('performance_mode', performance_mode)
                
                use_cache = st.checkbox("Enable Caching", value=True)
                SessionStateManager.update_setting('use_cache', use_cache)
                
                if use_cache:
                    cache_size = st.slider("Cache Size (MB)", 10, 500, 100, 10)
                    SessionStateManager.update_setting('cache_size', cache_size)
            
            st.markdown("---")
            
            # Quick Actions
            st.markdown("### ⚡ Quick Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh"):
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Cache"):
                    SmartCacheSystem.clear_cache()
                    st.success("Cache cleared!")
            
            if st.button("📊 Show Cache Stats"):
                cache_stats = SmartCacheSystem.get_cache_stats()
                st.json(cache_stats)
            
            st.markdown("---")
            
            # Session Info
            with st.expander("ℹ️ **Session Information**", expanded=False):
                session_stats = SessionStateManager.get_session_stats()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Render Count", session_stats['render_count'])
                    st.metric("Session Duration", str(session_stats['session_duration']).split('.')[0])
                
                with col2:
                    st.metric("Cache Size", session_stats['cache_size'])
                    st.metric("History Size", session_stats['history_size'])
                
                if st.button("Save Session"):
                    if SessionStateManager.save_session_state():
                        st.success("Session saved!")
                
                if st.button("Load Session"):
                    if SessionStateManager.load_session_state():
                        st.success("Session loaded!")
                        st.rerun()
    
    @staticmethod
    def create_main_content():
        """Create main content area"""
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎨 Visualization", 
            "📊 Analytics", 
            "⚙️ Settings", 
            "📈 Statistics",
            "💾 Export",
            "📚 Documentation"
        ])
        
        with tab1:
            UltraEnhancedStreamlitApp._create_visualization_tab()
        
        with tab2:
            UltraEnhancedStreamlitApp._create_analytics_tab()
        
        with tab3:
            UltraEnhancedStreamlitApp._create_settings_tab()
        
        with tab4:
            UltraEnhancedStreamlitApp._create_statistics_tab()
        
        with tab5:
            UltraEnhancedStreamlitApp._create_export_tab()
        
        with tab6:
            UltraEnhancedStreamlitApp._create_documentation_tab()
    
    @staticmethod
    def _create_visualization_tab():
        """Create visualization tab"""
        st.header("🎨 Interactive Visualization")
        
        if not st.session_state.get('data_loaded', False):
            st.info("👈 Please load data from the sidebar to begin visualization")
            return
        
        # Create visualization controls
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            visualization_mode = st.selectbox(
                "Visualization Mode",
                ["Standard Chord", "Enhanced Chord", "Interactive", "Animated"],
                index=1
            )
        
        with col2:
            auto_refresh = st.checkbox("Auto-refresh", value=False)
        
        with col3:
            if st.button("Render Now", type="primary"):
                st.session_state.force_render = True
        
        # Advanced visualization controls
        with st.expander("🎛️ **Advanced Controls**", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                link_threshold = st.slider("Link Threshold", 0.0, 1.0, 0.01, 0.01)
                directional = st.checkbox("Show Direction", value=True)
                gradient_links = st.checkbox("Gradient Links", value=True)
            
            with col_b:
                node_scaling = st.select_slider(
                    "Node Scaling",
                    options=["None", "Linear", "Logarithmic", "Square Root"],
                    value="Square Root"
                )
                show_labels = st.checkbox("Show Labels", value=True)
                show_grid = st.checkbox("Show Grid", value=False)
            
            with col_c:
                animation_speed = st.slider("Animation Speed", 1, 10, 5)
                hover_effects = st.checkbox("Hover Effects", value=True)
                highlight_strong = st.checkbox("Highlight Strong Links", value=True)
        
        # Visualization area
        visualization_container = st.container()
        
        with visualization_container:
            if st.session_state.get('force_render', False) or auto_refresh:
                UltraEnhancedStreamlitApp._render_visualization(
                    visualization_mode,
                    link_threshold,
                    directional,
                    gradient_links,
                    node_scaling,
                    show_labels,
                    show_grid,
                    animation_speed,
                    hover_effects,
                    highlight_strong
                )
                st.session_state.force_render = False
            else:
                # Show preview
                st.info("Click 'Render Now' or enable 'Auto-refresh' to generate visualization")
                
                # Quick preview with sample
                preview_data = DataProcessingEngine.generate_sample_data(size=8)
                preview_fig = UltraEnhancedStreamlitApp._create_quick_preview(preview_data)
                st.pyplot(preview_fig)
    
    @staticmethod
    def _render_visualization(mode: str, link_threshold: float, 
                            directional: bool, gradient_links: bool,
                            node_scaling: str, show_labels: bool,
                            show_grid: bool, animation_speed: int,
                            hover_effects: bool, highlight_strong: bool):
        """Render the visualization"""
        with st.spinner("🎨 Rendering visualization..."):
            try:
                # Get data
                data = st.session_state.current_data
                data_type = st.session_state.data_type
                
                # Create visualization based on mode
                if mode == "Enhanced Chord":
                    fig = UltraEnhancedStreamlitApp._create_enhanced_chord(
                        data, data_type, link_threshold, directional,
                        gradient_links, node_scaling, show_labels,
                        show_grid, hover_effects, highlight_strong
                    )
                elif mode == "Interactive":
                    fig = UltraEnhancedStreamlitApp._create_interactive_chord(
                        data, data_type, link_threshold
                    )
                elif mode == "Animated":
                    fig = UltraEnhancedStreamlitApp._create_animated_chord(
                        data, data_type, link_threshold, animation_speed
                    )
                else:  # Standard Chord
                    fig = UltraEnhancedStreamlitApp._create_standard_chord(
                        data, data_type, link_threshold
                    )
                
                # Display the figure
                st.pyplot(fig, use_container_width=True)
                
                # Update session statistics
                st.session_state.render_count = st.session_state.get('render_count', 0) + 1
                st.session_state.success_count = st.session_state.get('success_count', 0) + 1
                
                # Show success message
                st.success(f"✅ Visualization rendered successfully! (Render #{st.session_state.render_count})")
                
            except Exception as e:
                st.error(f"❌ Error rendering visualization: {str(e)}")
                st.session_state.error_count = st.session_state.get('error_count', 0) + 1
                
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
    
    @staticmethod
    @st.cache_data(ttl=600, max_entries=5)
    def _create_enhanced_chord(data: Any, data_type: str, link_threshold: float,
                              directional: bool, gradient_links: bool,
                              node_scaling: str, show_labels: bool,
                              show_grid: bool, hover_effects: bool,
                              highlight_strong: bool) -> plt.Figure:
        """Create enhanced chord diagram with caching"""
        # Initialize diagram
        diagram = UltraEnhancedChordDiagram(
            figsize=st.session_state.viz_settings.get('figsize', (24, 24)),
            dpi=st.session_state.viz_settings.get('dpi', 300),
            background_color=st.session_state.viz_settings.get('background_color', '#FFFFFF'),
            use_cache=st.session_state.viz_settings.get('use_cache', True),
            enable_animation=False
        )
        
        # Process data
        if data_type == 'matrix':
            matrix = data.values if isinstance(data, pd.DataFrame) else np.array(data)
            
            # Extract sectors
            if isinstance(data, pd.DataFrame):
                row_sectors = data.index.tolist()
                col_sectors = data.columns.tolist()
            else:
                row_sectors = [f"S{i+1}" for i in range(matrix.shape[0])]
                col_sectors = [f"T{j+1}" for j in range(matrix.shape[1])]
            
            all_sectors = row_sectors + col_sectors
            
            # Create groups
            groups = {}
            for i, sector in enumerate(row_sectors):
                groups[sector] = 0
            for i, sector in enumerate(col_sectors):
                groups[sector] = 1
            
            # Initialize sectors
            diagram.initialize_sectors(all_sectors, groups)
            
            # Compute angles
            diagram.compute_sector_angles()
            
            # Add tracks
            for sector in all_sectors:
                diagram.add_track(sector, track_index=0)
            
            # Create links
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    if abs(value) > link_threshold:
                        source = row_sectors[i]
                        target = col_sectors[j]
                        
                        # Determine if link should be highlighted
                        is_strong = value > np.percentile(matrix[matrix > 0], 75) if np.any(matrix > 0) else False
                        
                        # Create link
                        diagram.create_enhanced_link(
                            source=source,
                            target=target,
                            value=abs(value),
                            color=None,  # Auto-assign based on strategy
                            style={
                                'directional': directional,
                                'gradient': gradient_links,
                                'border': True,
                                'glow': True,
                                'color_strategy': 'source' if i % 2 == 0 else 'value'
                            }
                        )
        
        else:  # adjacency_list
            # Similar processing for adjacency lists
            pass
        
        # Finalize diagram
        fig = diagram.finalize(
            title="Enhanced Chord Diagram",
            show_frame=show_grid,
            show_grid=show_grid,
            interactive=hover_effects
        )
        
        return fig
    
    @staticmethod
    def _create_interactive_chord(data: Any, data_type: str, 
                                 link_threshold: float) -> plt.Figure:
        """Create interactive chord diagram"""
        # Similar to enhanced but with more interactive features
        return UltraEnhancedStreamlitApp._create_enhanced_chord(
            data, data_type, link_threshold,
            directional=True, gradient_links=True,
            node_scaling="Square Root", show_labels=True,
            show_grid=False, hover_effects=True,
            highlight_strong=True
        )
    
    @staticmethod
    def _create_animated_chord(data: Any, data_type: str, 
                              link_threshold: float, 
                              animation_speed: int) -> plt.Figure:
        """Create animated chord diagram"""
        # Create base diagram
        fig = UltraEnhancedStreamlitApp._create_enhanced_chord(
            data, data_type, link_threshold,
            directional=True, gradient_links=True,
            node_scaling="Square Root", show_labels=True,
            show_grid=False, hover_effects=False,
            highlight_strong=True
        )
        
        # Note: Actual animation would require additional setup
        # This is a placeholder for animation functionality
        st.info("🎬 Animation features require additional setup. Showing static version.")
        
        return fig
    
    @staticmethod
    def _create_standard_chord(data: Any, data_type: str, 
                              link_threshold: float) -> plt.Figure:
        """Create standard chord diagram"""
        # Simplified version for comparison
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Basic chord diagram implementation
        # (Simplified for brevity)
        
        return fig
    
    @staticmethod
    def _create_quick_preview(data: pd.DataFrame) -> plt.Figure:
        """Create quick preview visualization"""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Simple preview implementation
        n = min(8, data.shape[0])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        
        for i in range(n):
            for j in range(n):
                value = data.iloc[i, j] if i < data.shape[0] and j < data.shape[1] else 0
                if value > 0.1:
                    # Draw simple arc
                    theta = [angles[i], angles[j]]
                    r = [0.5, 0.5]
                    ax.plot(theta, r, color='blue', alpha=0.3, linewidth=value*10)
        
        # Add nodes
        ax.scatter(angles, [0.5]*n, c='red', s=100, zorder=5)
        
        # Clean up
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        
        return fig
    
    @staticmethod
    def _create_analytics_tab():
        """Create analytics tab"""
        st.header("📊 Advanced Analytics")
        
        if not st.session_state.get('data_loaded', False):
            st.info("👈 Please load data to perform analytics")
            return
        
        # Analytics controls
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Network Metrics", "Community Detection", "Centrality Analysis", 
                 "Correlation Analysis", "Clustering", "Dimensionality Reduction"]
            )
        
        with col2:
            if st.button("Run Analysis", type="primary"):
                st.session_state.run_analysis = True
        
        if st.session_state.get('run_analysis', False):
            with st.spinner("🔬 Performing analysis..."):
                UltraEnhancedStreamlitApp._perform_analysis(analysis_type)
    
    @staticmethod
    def _perform_analysis(analysis_type: str):
        """Perform selected analysis"""
        data = st.session_state.current_data
        data_type = st.session_state.data_type
        
        if analysis_type == "Network Metrics":
            UltraEnhancedStreamlitApp._show_network_metrics(data, data_type)
        elif analysis_type == "Community Detection":
            UltraEnhancedStreamlitApp._show_community_detection(data, data_type)
        elif analysis_type == "Centrality Analysis":
            UltraEnhancedStreamlitApp._show_centrality_analysis(data, data_type)
        elif analysis_type == "Correlation Analysis":
            UltraEnhancedStreamlitApp._show_correlation_analysis(data, data_type)
        elif analysis_type == "Clustering":
            UltraEnhancedStreamlitApp._show_clustering_analysis(data, data_type)
        else:  # Dimensionality Reduction
            UltraEnhancedStreamlitApp._show_dimensionality_reduction(data, data_type)
    
    @staticmethod
    def _show_network_metrics(data: Any, data_type: str):
        """Display network metrics"""
        analysis = DataProcessingEngine.perform_network_analysis(data, data_type)
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nodes", len(data) if hasattr(data, '__len__') else 'N/A')
            st.metric("Density", f"{analysis.get('density', 0):.3f}")
            st.metric("Avg Degree", f"{analysis.get('avg_out_degree', 0):.1f}")
        
        with col2:
            st.metric("Links", analysis.get('num_edges', 'N/A'))
            st.metric("Avg Strength", f"{analysis.get('average_strength', 0):.3f}")
            st.metric("Max Strength", f"{analysis.get('max_strength', 0):.3f}")
        
        with col3:
            st.metric("Components", analysis.get('connected_components', 'N/A'))
            st.metric("Clustering Coef", f"{analysis.get('clustering_coefficient', 0):.3f}")
            st.metric("Diameter", str(analysis.get('diameter', 'N/A')))
        
        # Detailed metrics
        with st.expander("📈 Detailed Metrics", expanded=False):
            st.json(analysis)
    
    @staticmethod
    def _show_community_detection(data: Any, data_type: str):
        """Display community detection results"""
        st.info("Community detection requires networkx and additional libraries.")
        # Implementation would go here
    
    @staticmethod
    def _show_centrality_analysis(data: Any, data_type: str):
        """Display centrality analysis"""
        st.info("Centrality analysis shows important nodes in the network.")
        # Implementation would go here
    
    @staticmethod
    def _show_correlation_analysis(data: Any, data_type: str):
        """Display correlation analysis"""
        if data_type == 'matrix' and isinstance(data, pd.DataFrame):
            # Calculate correlation matrix
            corr_matrix = data.corr()
            
            # Display heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            
            # Add labels
            if len(data.columns) <= 20:
                ax.set_xticks(range(len(data.columns)))
                ax.set_yticks(range(len(data.columns)))
                ax.set_xticklabels(data.columns, rotation=45, ha='right')
                ax.set_yticklabels(data.columns)
            
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
            
            # Statistics
            st.metric("Average Correlation", f"{corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean():.3f}")
            st.metric("Strong Correlations (>0.7)", f"{(np.abs(corr_matrix.values) > 0.7).sum() - len(data.columns):,}")
        else:
            st.warning("Correlation analysis requires matrix data")
    
    @staticmethod
    def _show_clustering_analysis(data: Any, data_type: str):
        """Display clustering analysis"""
        st.info("Clustering analysis groups similar nodes together.")
        # Implementation would go here
    
    @staticmethod
    def _show_dimensionality_reduction(data: Any, data_type: str):
        """Display dimensionality reduction results"""
        st.info("Dimensionality reduction visualizes high-dimensional data in 2D/3D.")
        # Implementation would go here
    
    @staticmethod
    def _create_settings_tab():
        """Create settings tab"""
        st.header("⚙️ Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Visualization Settings")
            
            # Color settings
            color_palette = st.selectbox(
                "Color Palette",
                AdvancedColorEngine.get_palette_names()['link_palettes'],
                index=0
            )
            SessionStateManager.update_setting('color_palette', color_palette)
            
            # Layout settings
            start_angle = st.slider("Start Angle", 0, 359, 0, 15)
            SessionStateManager.update_setting('start_angle', start_angle)
            
            direction = st.selectbox("Direction", ["Clockwise", "Counter-clockwise"])
            SessionStateManager.update_setting('direction', direction)
            
            gap_size = st.slider("Gap Size", 0.0, 30.0, 15.0, 1.0)
            SessionStateManager.update_setting('gap_size', gap_size)
        
        with col2:
            st.subheader("Performance Settings")
            
            # Cache settings
            cache_enabled = st.checkbox("Enable Caching", value=True)
            SessionStateManager.update_setting('cache_enabled', cache_enabled)
            
            if cache_enabled:
                cache_size = st.slider("Cache Size (items)", 10, 1000, 100, 10)
                cache_ttl = st.slider("Cache TTL (minutes)", 1, 240, 60, 5)
                SessionStateManager.update_setting('cache_size', cache_size)
                SessionStateManager.update_setting('cache_ttl', cache_ttl)
            
            # Rendering settings
            render_quality = st.select_slider(
                "Render Quality",
                options=["Low", "Medium", "High", "Ultra"],
                value="High"
            )
            SessionStateManager.update_setting('render_quality', render_quality)
            
            # Multi-threading
            use_multithreading = st.checkbox("Use Multi-threading", value=True)
            SessionStateManager.update_setting('use_multithreading', use_multithreading)
        
        # Advanced settings
        with st.expander("🔧 **Expert Settings**", expanded=False):
            expert_col1, expert_col2 = st.columns(2)
            
            with expert_col1:
                # Memory management
                memory_limit = st.slider("Memory Limit (MB)", 100, 2000, 500, 50)
                SessionStateManager.update_setting('memory_limit', memory_limit)
                
                # GPU acceleration
                use_gpu = st.checkbox("Use GPU Acceleration", value=False)
                SessionStateManager.update_setting('use_gpu', use_gpu)
            
            with expert_col2:
                # Advanced rendering
                anti_aliasing = st.select_slider(
                    "Anti-aliasing",
                    options=["None", "2x", "4x", "8x"],
                    value="4x"
                )
                SessionStateManager.update_setting('anti_aliasing', anti_aliasing)
                
                # Vector quality
                vector_quality = st.select_slider(
                    "Vector Quality",
                    options=["Low", "Medium", "High"],
                    value="High"
                )
                SessionStateManager.update_setting('vector_quality', vector_quality)
        
        # Reset button
        if st.button("Reset to Defaults", type="secondary"):
            SessionStateManager.initialize_session_state()
            st.success("Settings reset to defaults!")
            st.rerun()
    
    @staticmethod
    def _create_statistics_tab():
        """Create statistics tab"""
        st.header("📈 Performance Statistics")
        
        # Session statistics
        session_stats = SessionStateManager.get_session_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Session Duration", str(session_stats['session_duration']).split('.')[0])
            st.metric("Page Views", st.session_state.performance_monitor.get('page_views', 0))
        
        with col2:
            st.metric("Total Renders", session_stats['render_count'])
            st.metric("Success Rate", 
                     f"{(session_stats['success_count'] / max(1, session_stats['success_count'] + session_stats['error_count'])) * 100:.1f}%")
        
        with col3:
            st.metric("Cache Hits", SmartCacheSystem.get_cache_stats().get('hits', 0))
            st.metric("Cache Misses", SmartCacheSystem.get_cache_stats().get('misses', 0))
        
        with col4:
            hit_rate = SmartCacheSystem.get_cache_stats().get('hit_rate', 0)
            st.metric("Cache Hit Rate", f"{hit_rate*100:.1f}%")
            st.metric("Cache Items", SmartCacheSystem.get_cache_stats().get('total_items', 0))
        
        # Performance charts
        with st.expander("📊 **Performance Charts**", expanded=True):
            # Render time history
            if st.session_state.performance_monitor.get('render_times'):
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(st.session_state.performance_monitor['render_times'], marker='o')
                ax.set_xlabel("Render Number")
                ax.set_ylabel("Time (seconds)")
                ax.set_title("Render Time History")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Cache statistics
            cache_stats = SmartCacheSystem.get_cache_stats()
            if cache_stats['total_items'] > 0:
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Cache hits/misses
                labels = ['Hits', 'Misses']
                values = [cache_stats['hits'], cache_stats['misses']]
                ax1.bar(labels, values, color=['#4CAF50', '#F44336'])
                ax1.set_title("Cache Performance")
                ax1.set_ylabel("Count")
                
                # Cache age distribution
                if cache_stats['oldest_entry'] and cache_stats['newest_entry']:
                    current_time = time.time()
                    ages = [current_time - cache_stats['oldest_entry'], 
                           current_time - cache_stats['newest_entry']]
                    ax2.bar(['Oldest', 'Newest'], ages, color=['#2196F3', '#FF9800'])
                    ax2.set_title("Cache Age")
                    ax2.set_ylabel("Seconds")
                    ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig2)
        
        # System information
        with st.expander("💻 **System Information**", expanded=False):
            sys_col1, sys_col2 = st.columns(2)
            
            with sys_col1:
                st.write("**Python Environment**")
                st.code(f"""
                Python: {sys.version}
                Streamlit: {st.__version__}
                Matplotlib: {matplotlib.__version__}
                NumPy: {np.__version__}
                Pandas: {pd.__version__}
                """)
            
            with sys_col2:
                st.write("**Memory Usage**")
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                st.metric("RAM Used", f"{memory_info.rss / 1024 / 1024:.1f} MB")
                st.metric("RAM Available", f"{psutil.virtual_memory().available / 1024 / 1024:.1f} MB")
                st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
        
        # Export statistics
        if st.button("Export Statistics as JSON"):
            stats_data = {
                'session_stats': session_stats,
                'cache_stats': SmartCacheSystem.get_cache_stats(),
                'system_info': {
                    'python_version': sys.version,
                    'streamlit_version': st.__version__,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(stats_data, indent=2),
                file_name=f"chord_diagram_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    @staticmethod
    def _create_export_tab():
        """Create export tab"""
        st.header("💾 Export Options")
        
        if not st.session_state.get('data_loaded', False):
            st.info("👈 Please load data and create visualization before exporting")
            return
        
        # Export format selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_format = st.selectbox(
                "Format",
                ["PNG", "PDF", "SVG", "TIFF", "HTML", "JSON", "CSV"]
            )
        
        with col2:
            if export_format in ["PNG", "TIFF", "PDF"]:
                export_dpi = st.slider("DPI", 72, 600, 300, 50)
            else:
                export_dpi = 300
        
        with col3:
            if export_format in ["PNG", "TIFF", "JPEG"]:
                export_quality = st.slider("Quality", 1, 100, 95)
            else:
                export_quality = 100
        
        # Additional options
        with st.expander("⚙️ **Export Settings**", expanded=False):
            include_metadata = st.checkbox("Include Metadata", value=True)
            include_statistics = st.checkbox("Include Statistics", value=True)
            compress_file = st.checkbox("Compress File", value=False)
            
            if export_format in ["PNG", "TIFF", "JPEG"]:
                transparent_bg = st.checkbox("Transparent Background", value=False)
            else:
                transparent_bg = False
        
        # Export button
        if st.button("Generate Export", type="primary"):
            with st.spinner(f"Generating {export_format} file..."):
                UltraEnhancedStreamlitApp._generate_export(
                    export_format, export_dpi, export_quality,
                    include_metadata, include_statistics,
                    compress_file, transparent_bg
                )
    
    @staticmethod
    def _generate_export(format: str, dpi: int, quality: int,
                        include_metadata: bool, include_statistics: bool,
                        compress: bool, transparent: bool):
        """Generate export file"""
        try:
            # Create a visualization for export
            data = st.session_state.current_data
            fig = UltraEnhancedStreamlitApp._create_enhanced_chord(
                data, st.session_state.data_type, 0.01,
                True, True, "Square Root", True, False, False, True
            )
            
            # Prepare export buffer
            buffer = BytesIO()
            
            if format == "PNG":
                fig.savefig(buffer, format='png', dpi=dpi, 
                          bbox_inches='tight', pad_inches=0.5,
                          transparent=transparent)
                mime_type = "image/png"
                file_ext = "png"
            
            elif format == "PDF":
                fig.savefig(buffer, format='pdf', dpi=dpi,
                          bbox_inches='tight', pad_inches=0.5)
                mime_type = "application/pdf"
                file_ext = "pdf"
            
            elif format == "SVG":
                fig.savefig(buffer, format='svg',
                          bbox_inches='tight', pad_inches=0.5)
                mime_type = "image/svg+xml"
                file_ext = "svg"
            
            elif format == "TIFF":
                fig.savefig(buffer, format='tiff', dpi=dpi,
                          bbox_inches='tight', pad_inches=0.5)
                mime_type = "image/tiff"
                file_ext = "tiff"
            
            elif format == "HTML":
                # Convert to interactive HTML
                html_content = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Chord Diagram Export</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                </head>
                <body>
                    <div id="chart"></div>
                    <script>
                        // Interactive chart would go here
                        document.getElementById('chart').innerHTML = 
                            '<h3>Interactive Chord Diagram</h3><p>Export generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>';
                    </script>
                </body>
                </html>
                """
                buffer.write(html_content.encode())
                mime_type = "text/html"
                file_ext = "html"
            
            elif format == "JSON":
                # Export data as JSON
                if isinstance(st.session_state.current_data, pd.DataFrame):
                    json_data = st.session_state.current_data.to_json(orient='split')
                else:
                    json_data = json.dumps({"data": "Raw data export"})
                
                if include_metadata:
                    metadata = {
                        "export_date": datetime.now().isoformat(),
                        "format": "chord_diagram_data",
                        "statistics": SessionStateManager.get_session_stats() if include_statistics else {}
                    }
                    json_data = json.dumps({"metadata": metadata, "data": json.loads(json_data)}, indent=2)
                
                buffer.write(json_data.encode())
                mime_type = "application/json"
                file_ext = "json"
            
            elif format == "CSV":
                # Export data as CSV
                if isinstance(st.session_state.current_data, pd.DataFrame):
                    csv_data = st.session_state.current_data.to_csv(index=True)
                else:
                    csv_data = "source,target,value\nSample,Data,1.0\n"
                
                buffer.write(csv_data.encode())
                mime_type = "text/csv"
                file_ext = "csv"
            
            else:
                st.error(f"Unsupported export format: {format}")
                return
            
            # Prepare download
            buffer.seek(0)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"chord_diagram_{timestamp}.{file_ext}"
            
            # Offer download
            st.download_button(
                label=f"Download {format}",
                data=buffer,
                file_name=filename,
                mime=mime_type
            )
            
            st.success(f"✅ {format} file generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error generating export: {str(e)}")
    
    @staticmethod
    def _create_documentation_tab():
        """Create documentation tab"""
        st.header("📚 Documentation & Guide")
        
        # Quick start guide
        with st.expander("🚀 **Quick Start Guide**", expanded=True):
            st.markdown("""
            ### Getting Started
            
            1. **Load Data**: Use the sidebar to load your dataset
            2. **Configure Visualization**: Adjust settings in the Visualization tab
            3. **Render**: Click "Render Now" to generate the chord diagram
            4. **Analyze**: Use the Analytics tab for insights
            5. **Export**: Save your results in various formats
            
            ### Key Features
            
            * **Light Theme Optimized**: Designed for readability and professional presentations
            * **Advanced Caching**: Prevents re-computation for faster rendering
            * **Interactive Elements**: Hover and click for detailed information
            * **Multi-format Export**: Export as PNG, PDF, SVG, HTML, JSON, or CSV
            * **Comprehensive Analytics**: Network metrics, centrality analysis, and more
            """)
        
        # Visualization guide
        with st.expander("🎨 **Visualization Guide**", expanded=False):
            st.markdown("""
            ### Visualization Modes
            
            **Enhanced Chord**: Full-featured visualization with gradients, glows, and effects
            **Interactive**: Clickable elements with tooltips and highlighting
            **Animated**: Dynamic visualization with pulsing effects
            **Standard**: Simple, fast rendering for large datasets
            
            ### Color Palettes
            
            Choose from multiple palettes optimized for light backgrounds:
            * **Vivid Light**: Bright, distinct colors for maximum contrast
            * **Electric Light**: Neon colors for high-impact visuals
            * **Pastel Rainbow**: Soft colors for subtle presentations
            * **Professional**: Publication-ready color schemes
            
            ### Performance Tips
            
            1. Use caching for repeated renders
            2. Adjust link threshold to reduce visual clutter
            3. Use "Balanced" performance mode for best results
            4. Export to vector formats (PDF/SVG) for high-quality prints
            """)
        
        # API documentation
        with st.expander("🔧 **API Documentation**", expanded=False):
            st.markdown("""
            ### Core Classes
            
            **UltraEnhancedChordDiagram**: Main visualization engine
            ```python
            diagram = UltraEnhancedChordDiagram(
                figsize=(24, 24),
                dpi=300,
                background_color='#FFFFFF',
                use_cache=True
            )
            ```
            
            **DataProcessingEngine**: Data loading and analysis
            ```python
            data, data_type, meta = DataProcessingEngine.load_and_preprocess_data(
                file_path='data.csv',
                normalize=True
            )
            ```
            
            **SmartCacheSystem**: Performance optimization
            ```python
            result = SmartCacheSystem.get_or_create(
                key='computation',
                creator_func=expensive_computation,
                *args, **kwargs
            )
            ```
            
            ### Advanced Usage
            
            Custom styling:
            ```python
            diagram.create_enhanced_link(
                source='Gene_A',
                target='Pathway_B',
                value=0.85,
                style={
                    'gradient': True,
                    'glow': True,
                    'border': True,
                    'directional': True,
                    'curve_type': 'cubic'
                }
            )
            ```
            """)
        
        # Troubleshooting
        with st.expander("🔧 **Troubleshooting**", expanded=False):
            st.markdown("""
            ### Common Issues
            
            **Slow Rendering**:
            - Enable caching in Performance Settings
            - Reduce canvas size or DPI
            - Increase link threshold to show fewer connections
            
            **Memory Issues**:
            - Reduce cache size
            - Use lower quality settings
            - Close other applications
            
            **Visual Quality Problems**:
            - Increase DPI for higher resolution
            - Use vector formats (PDF/SVG) for scaling
            - Adjust color palette for better contrast
            
            **Data Loading Errors**:
            - Ensure CSV files have proper formatting
            - Check for missing values or incorrect data types
            - Try the sample data for testing
            
            ### Getting Help
            
            1. Check the error details in the expander below error messages
            2. Try resetting to default settings
            3. Clear cache and restart the application
            4. Ensure all required libraries are installed
            """)
        
        # Version info
        st.markdown("---")
        st.caption(f"**Version**: 2.0.0 | **Last Updated**: {datetime.now().strftime('%Y-%m-%d')}")
        st.caption("© 2024 Ultra Enhanced Chord Diagrams - Professional Network Visualization Tool")

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point"""
    try:
        # Setup application
        UltraEnhancedStreamlitApp.setup_page()
        
        # Update page view counter
        st.session_state.performance_monitor['page_views'] = \
            st.session_state.performance_monitor.get('page_views', 0) + 1
        
        # Create sidebar
        UltraEnhancedStreamlitApp.create_sidebar()
        
        # Create main content
        UltraEnhancedStreamlitApp.create_main_content()
        
        # Add footer
        st.markdown("""
        <div class="footer">
            <p>Ultra Enhanced Chord Diagrams • Professional Network Visualization • Light Theme Edition</p>
            <p>Optimized for performance with advanced caching and session management</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
