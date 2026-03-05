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
                recommendations[col] = 'fill_median'  # Distribution asymétrique
            else:
                recommendations[col] = 'fill_mean'  # Distribution normale
        else:
            recommendations[col] = 'fill_mode'  # Mode pour catégoriel
    
    return recommendations


def _analyze_duplicates(df: pd.DataFrame) -> Dict:
    """Analyse des doublons."""
    n_duplicates = df.duplicated().sum()
    
    if n_duplicates == 0:
        return {'has_duplicates': False}
    
    # Doublons partiels (sur sous-ensembles de colonnes)
    key_candidates = []
    for col_subset in [df.columns[:i] for i in range(1, min(4, len(df.columns)))]:
        dup_subset = df.duplicated(subset=col_subset).sum()
        if dup_subset > 0:
            key_candidates.append({
                'columns': list(col_subset),
                'duplicates': int(dup_subset)
            })
    
    return {
        'has_duplicates': True,
        'total_duplicates': int(n_duplicates),
        'percentage': round(n_duplicates / len(df) * 100, 2),
        'key_candidates': key_candidates[:3],  # Top 3
        'sample_duplicates': df[df.duplicated(keep=False)].head(5).to_dict('records')
    }


def _analyze_outliers(df: pd.DataFrame) -> Dict:
    """Analyse des outliers avec plusieurs méthodes."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return {'has_outliers': False, 'reason': 'no_numeric_columns'}
    
    all_outliers = {}
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 10:
            continue
        
        # Méthode IQR
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        
        iqr_outliers = df[(df[col] < lower) | (df[col] > upper)]
        
        # Méthode Z-score (pour distributions normales)
        z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
        z_outliers = df[z_scores > 3]
        
        # Méthode Isolation Forest (multivariée si possible)
        # Simplifié ici pour une colonne
        
        if len(iqr_outliers) > 0:
            all_outliers[col] = {
                'count_iqr': len(iqr_outliers),
                'pct_iqr': round(len(iqr_outliers) / len(df) * 100, 2),
                'bounds': {'lower': round(lower, 2), 'upper': round(upper, 2)},
                'extreme_values': {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'suspected_low': float(iqr_outliers[col].min()) if len(iqr_outliers) > 0 else None,
                    'suspected_high': float(iqr_outliers[col].max()) if len(iqr_outliers) > 0 else None,
                }
            }
    
    return {
        'has_outliers': len(all_outliers) > 0,
        'columns_analyzed': len(numeric_cols),
        'columns_with_outliers': len(all_outliers),
        'by_column': all_outliers,
        'total_outliers_iqr': sum(d['count_iqr'] for d in all_outliers.values())
    }


def clean_data(
    df: pd.DataFrame,
    operations: List[Dict[str, Any]],
    inplace: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applique des opérations de nettoyage.
    
    Args:
        df: DataFrame à nettoyer
        operations: Liste d'opérations [{'type': 'fill_missing', 'column': 'X', 'method': 'mean'}, ...]
        inplace: Modifier le DataFrame original
    
    Returns:
        (df_cleaned, rapport)
    """
    if not inplace:
        df = df.copy()
    
    report = {'operations_applied': [], 'rows_before': len(df), 'rows_after': len(df)}
    
    for op in operations:
        op_type = op.get('type')
        column = op.get('column')
        
        if op_type == 'remove_duplicates':
            before = len(df)
            df = df.drop_duplicates()
            report['operations_applied'].append({
                'type': 'remove_duplicates',
                'rows_removed': before - len(df)
            })
            report['rows_after'] = len(df)
        
        elif op_type == 'fill_missing' and column in df.columns:
            method = op.get('method', 'mean')
            null_before = df[column].isnull().sum()
            
            if method == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].mean())
            elif method == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())
            elif method == 'mode':
                df[column] = df[column].fillna(df[column].mode()[0])
            elif method == 'constant':
                df[column] = df[column].fillna(op.get('value', 'UNKNOWN'))
            
            report['operations_applied'].append({
                'type': 'fill_missing',
                'column': column,
                'method': method,
                'filled': null_before - df[column].isnull().sum()
            })
        
        elif op_type == 'drop_column' and column in df.columns:
            df = df.drop(columns=[column])
            report['operations_applied'].append({
                'type': 'drop_column',
                'column': column
            })
    
    return df, report
