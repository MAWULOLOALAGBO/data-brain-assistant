"""
Module d'exploration de données.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Types d'exploration disponibles
EXPLORATION_AVAILABLE = {
    'profile': 'Profil complet du dataset',
    'describe': 'Statistiques descriptives',
    'correlation': 'Matrice de corrélation',
    'dtypes': 'Types de données',
    'head': 'Premières lignes',
    'tail': 'Dernières lignes',
    'sample': 'Échantillon aléatoire',
}


def explore_data(df: pd.DataFrame, exploration_type: str = 'profile', **kwargs) -> Dict[str, Any]:
    """
    Explore le dataset selon le type demandé.
    
    Args:
        df: DataFrame à explorer
        exploration_type: Type d'exploration (voir EXPLORATION_AVAILABLE)
        **kwargs: Options spécifiques
    
    Returns:
        Résultat structuré de l'exploration
    """
    
    if exploration_type == 'profile':
        return _generate_profile(df)
    elif exploration_type == 'describe':
        return _generate_describe(df, kwargs.get('include', 'all'))
    elif exploration_type == 'correlation':
        return _generate_correlation(df, kwargs.get('method', 'pearson'))
    elif exploration_type == 'dtypes':
        return _generate_dtypes(df)
    elif exploration_type == 'head':
        return df.head(kwargs.get('n', 10))
    elif exploration_type == 'tail':
        return df.tail(kwargs.get('n', 10))
    elif exploration_type == 'sample':
        n = min(kwargs.get('n', 10), len(df))
        return df.sample(n)
    else:
        raise ValueError(f"Type d'exploration inconnu: {exploration_type}")


def _generate_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Génère un profil complet du dataset."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    profile = {
        'overview': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'datetime_columns': len(datetime_cols),
        },
        'column_details': {}
    }
    
    for col in df.columns:
        col_profile = {
            'type': str(df[col].dtype),
            'null_count': int(df[col].isnull().sum()),
            'null_pct': round(df[col].isnull().mean() * 100, 2),
            'unique_count': int(df[col].nunique()),
            'memory_kb': round(df[col].memory_usage(deep=True) / 1024, 2)
        }
        
        if col in numeric_cols:
            col_profile.update({
                'min': float(df[col].min()) if not df[col].isnull().all() else None,
                'max': float(df[col].max()) if not df[col].isnull().all() else None,
                'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                'std': float(df[col].std()) if not df[col].isnull().all() else None,
                'zeros': int((df[col] == 0).sum()),
                'negatives': int((df[col] < 0).sum()) if (df[col] < 0).any() else 0,
            })
        elif col in categorical_cols:
            top_values = df[col].value_counts().head(5).to_dict()
            col_profile.update({
                'top_5': {str(k): int(v) for k, v in top_values.items()},
                'is_boolean': df[col].nunique() == 2,
            })
        
        profile['column_details'][col] = col_profile
    
    # Détections automatiques
    profile['warnings'] = _detect_warnings(df, numeric_cols, categorical_cols)
    
    return profile


def _detect_warnings(df, numeric_cols, categorical_cols) -> List[Dict]:
    """Détecte des problèmes potentiels."""
    warnings = []
    
    # Colonnes quasi-vides
    for col in df.columns:
        null_pct = df[col].isnull().mean()
        if null_pct > 0.9:
            warnings.append({
                'type': 'almost_empty',
                'column': col,
                'severity': 'high',
                'message': f"{null_pct*100:.1f}% de valeurs manquantes"
            })
    
    # Colonnes constantes
    for col in df.columns:
        if df[col].nunique() == 1:
            warnings.append({
                'type': 'constant',
                'column': col,
                'severity': 'medium',
                'message': f"Valeur unique: {df[col].iloc[0]}"
            })
    
    # Colonnes avec beaucoup de catégories
    for col in categorical_cols:
        if df[col].nunique() > 1000:
            warnings.append({
                'type': 'high_cardinality',
                'column': col,
                'severity': 'medium',
                'message': f"{df[col].nunique()} catégories uniques"
            })
    
    # IDs potentiels (index)
    for col in df.columns:
        if 'id' in col.lower() and df[col].nunique() == len(df):
            warnings.append({
                'type': 'potential_id',
                'column': col,
                'severity': 'low',
                'message': "Colonne semblant être un identifiant unique"
            })
    
    return warnings


def _generate_describe(df: pd.DataFrame, include: str = 'all') -> pd.DataFrame:
    """Génère des statistiques descriptives enrichies."""
    desc = df.describe(include=include).transpose()
    
    # Ajouter des métriques supplémentaires
    if 'mean' in desc.columns:
        desc['cv'] = desc['std'] / desc['mean']  # Coefficient de variation
        desc['skewness'] = df.skew()
        desc['kurtosis'] = df.kurtosis()
        desc['iqr'] = desc['75%'] - desc['25%']
        desc['range'] = desc['max'] - desc['min']
        desc['null_pct'] = df.isnull().mean() * 100
    
    return desc.round(4)


def _generate_correlation(df: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
    """Génère une analyse de corrélation complète."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return {
            'error': 'Moins de 2 colonnes numériques pour la corrélation',
            'correlation_matrix': None,
            'strong_correlations': []
        }
    
    corr_matrix = numeric_df.corr(method=method)
    
    # Trouver les corrélations fortes
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.7:  # Seuil de corrélation forte
                strong_corrs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': round(val, 3),
                    'strength': 'très forte' if abs(val) > 0.9 else 'forte',
                    'direction': 'positive' if val > 0 else 'négative'
                })
    
    # Trier par force
    strong_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return {
        'method': method,
        'correlation_matrix': corr_matrix.round(3),
        'strong_correlations': strong_corrs[:10],  # Top 10
        'n_variables': len(numeric_df.columns)
    }


def _generate_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Génère un rapport détaillé sur les types de données."""
    info = []
    for col in df.columns:
        info.append({
            'column': col,
            'pandas_type': str(df[col].dtype),
            'logical_type': _infer_logical_type(df[col]),
            'nullable': df[col].isnull().any(),
            'unique_ratio': round(df[col].nunique() / len(df), 4) if len(df) > 0 else 0,
            'memory_usage_kb': round(df[col].memory_usage(deep=True) / 1024, 2),
            'suggested_type': _suggest_optimized_type(df[col])
        })
    
    return pd.DataFrame(info)


def _infer_logical_type(series: pd.Series) -> str:
    """Infère le type logique d'une série."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    elif pd.api.types.is_numeric_dtype(series):
        if series.nunique() == 2:
            return 'binary_numeric'
        return 'continuous' if series.nunique() > 20 else 'discrete'
    elif pd.api.types.is_categorical_dtype(series):
        return 'categorical'
    else:
        # Heuristiques pour texte
        if series.str.contains(r'^\d{4}-\d{2}-\d{2}', regex=True, na=False).mean() > 0.5:
            return 'likely_date'
        if series.str.contains(r'@', na=False).mean() > 0.3:
            return 'likely_email'
        if series.str.len().mean() > 100:
            return 'long_text'
        return 'short_text'


def _suggest_optimized_type(series: pd.Series) -> str:
    """Suggère un type optimisé pour la mémoire."""
    if pd.api.types.is_integer_dtype(series):
        return 'int32 or int16' if series.max() < 32767 else 'int64'
    elif pd.api.types.is_float_dtype(series):
        return 'float32' if series.max() < 1e38 else 'float64'
    elif pd.api.types.is_object_dtype(series) and series.nunique() / len(series) < 0.5:
        return 'category'
    return str(series.dtype)
