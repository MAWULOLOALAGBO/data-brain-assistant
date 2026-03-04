"""Tests pour le module parser"""

import pytest
import pandas as pd
from core.parser import (
    normalize_text, find_column_candidates, detect_intention,
    detect_groupby, parse_query, IntentionType, ActionType
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Age': [25, 30, 35],
        'User_ID': ['U1', 'U2', 'U3'],
        'Daily_Phone_Hours': [5.5, 3.2, 7.1],
        'Gender': ['M', 'F', 'M']
    })


def test_normalize_text():
    assert normalize_text("  Héllo  Wörld!!  ") == "hello world"


def test_find_column_candidates(sample_df):
    columns = list(sample_df.columns)
    
    # Correspondance exacte
    candidates = find_column_candidates("age", columns)
    assert candidates[0][0] == 'Age'
    assert candidates[0][1] > 0.8
    
    # Correspondance partielle
    candidates = find_column_candidates("phone hours", columns)
    assert any(col == 'Daily_Phone_Hours' for col, _ in candidates)


def test_detect_intention():
    # Statistique
    intent, action, conf = detect_intention("quelle est la moyenne")
    assert intent == IntentionType.STATISTIQUE
    assert action == ActionType.MOYENNE
    
    # Visualisation
    intent, action, conf = detect_intention("fais un histogramme")
    assert intent == IntentionType.VISUALISATION
    assert action == ActionType.HISTOGRAMME
    
    # Nettoyage
    intent, action, conf = detect_intention("valeurs manquantes")
    assert intent == IntentionType.NETTOYAGE


def test_detect_groupby(sample_df):
    columns = list(sample_df.columns)
    
    groupby = detect_groupby("moyenne par Gender", columns)
    assert groupby == 'Gender'
    
    groupby = detect_groupby("sum by User_ID", columns)
    assert groupby == 'User_ID'


def test_parse_query_complete(sample_df):
    plan = parse_query("moyenne de Age par Gender", sample_df)
    
    assert plan.intention == IntentionType.STATISTIQUE
    assert plan.action == ActionType.MOYENNE
    assert 'Age' in plan.target_columns
    assert plan.groupby_column == 'Gender'
    assert plan.confidence > 0.5
    assert "moyenne" in plan.explanation.lower()


def test_parse_query_visualization(sample_df):
    plan = parse_query("histogramme de Daily_Phone_Hours", sample_df)
    
    assert plan.intention == IntentionType.VISUALISATION
    assert plan.action == ActionType.HISTOGRAMME
    assert 'Daily_Phone_Hours' in plan.target_columns
