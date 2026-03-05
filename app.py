"""
Data Brain Assistant - Interface Streamlit finale
Architecture modulaire, sans exec(), 100% sécurisée
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Ajout du core au path
sys.path.insert(0, str(Path(__file__).parent))

from core.loader import load_file, infer_types, get_column_info
from core.parser import parse_query, PlanAction, IntentionType, ActionType
from core.validator import validate_plan, ValidationLevel
from core.executor import execute_action, ExecutionError
from core.actions import (
    STATS_AVAILABLE, VIZ_AVAILABLE, 
    CLEANING_AVAILABLE, EXPLORATION_AVAILABLE
)

# Configuration de la page
st.set_page_config(
    page_title="🧠 Data Brain Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .stPlotlyChart {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== INITIALISATION SESSION ====================

def init_session():
    """Initialise les variables de session."""
    defaults = {
        'data': None,
        'metadata': None,
        'history': [],
        'last_plan': None,
        'last_result': None,
        'show_advanced': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ==================== SIDEBAR ====================

def render_sidebar():
    """Rend la barre latérale."""
    with st.sidebar:
        st.title("🧠 Data Brain")
        st.markdown("*Assistant d'analyse universel*")
        st.divider()
        
        # Historique
        if st.session_state.history:
            st.header("📜 Historique")
            for i, item in enumerate(reversed(st.session_state.history
