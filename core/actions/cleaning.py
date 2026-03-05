"""
Module de nettoyage de données.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Actions de nettoyage disponibles
CLEANING_AVAILABLE = {
    'detect_missing': 'Détecter valeurs manquantes',
    'detect_duplicates': 'Détecter doublons',
    'detect_outliers': 'Détecter outliers (IQR & Isolation Forest)',
    'remove_duplicates': 'Supprimer les doublons',
    'fill_missing': 'Remplir valeurs manquantes',
}


def detect_issues(df: pd.DataFrame, issue_type: str = 'all') -> Dict[str, Any]:
    """
    Détecte les problèmes de qualité dans le dataset.
    
    Args:
        df: DataFrame à analyser
        issue_type: Type de problème ('missing', 'duplicates', 'outliers', 'all')
    
    Returns:
        Rapport détaillé des problèmes
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'issues_found': False,
        'details': {}
    }
    
    if issue_type in ['missing', 'all']:
        missing_report = _analyze_missing(df)
        if missing_report['has_missing']:
            report['issues_found'] = True
            report['details']['missing'] = missing_report
    
    if issue_type in ['duplicates', 'all']:
        dup_report = _analyze_duplicates(df)
        if dup_report['has_duplicates']:
            report['issues_found'] = True
            report['details']['duplicates'] = dup_report
    
    if issue_type in ['outliers', 'all']:
        outlier_report = _analyze_outliers(df)
        if outlier_report['has_outliers']:
            report['issues_found'] = True
            report['details']['outliers'] = outlier_report
    
    return report


def _analyze_missing(df: pd.DataFrame) -> Dict:
    """Analyse des valeurs manquantes."""
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    
    if len(null_counts) == 0:
        return {'has_missing': False}
    
    # Types de manquants
    missing_types = {}
    for col in null_counts.index:
        if pd.api.types.is_numeric_dtype(df[col]):
            missing_types[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            missing_types[col] = 'datetime'
        else:
            missing_types[col] = 'categorical'
    
    return {
        'has_missing': True,
        'total_missing': null_counts.sum(),
        'columns_affected': len(null_counts),
        'by_column': null_counts.to_dict(),
        'percentages': (null_counts / len(df) * 100).round(2).to_dict(),
        'types': missing_types,
        'recommendations': _recommend_missing_strategy(df, null_counts)
    }


def _recommend_missing_strategy(df: pd.DataFrame, null_counts: pd.Series) -> Dict[str, str]:
    """Recommande une stratégie de remplissage par colonne."""
    recommendations = {}
    
    for col in null_counts.index:
        null_pct = null_counts[col] / len(df)
        
        if null_pct > 0.5:
            recommendations[col] = 'drop_column'  # Trop de manquants
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].skew() > 1 or df[col].skew() < -1:
                recommendations[col] = 'fill_median'  # Distribution
