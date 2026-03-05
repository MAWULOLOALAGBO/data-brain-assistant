"""
Actions module for data analysis.
Each submodule provides specific functionality.
"""

from .stats import calculate_stats, STATS_AVAILABLE
from .viz import create_visualization, VIZ_AVAILABLE
from .cleaning import detect_issues, clean_data, CLEANING_AVAILABLE
from .exploration import explore_data, EXPLORATION_AVAILABLE

__all__ = [
    # Stats
    'calculate_stats', 'STATS_AVAILABLE',
    # Viz
    'create_visualization', 'VIZ_AVAILABLE',
    # Cleaning
    'detect_issues', 'clean_data', 'CLEANING_AVAILABLE',
    # Exploration
    'explore_data', 'EXPLORATION_AVAILABLE',
]
