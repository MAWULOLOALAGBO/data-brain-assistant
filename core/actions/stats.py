"""
Module de calculs statistiques.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
from core.parser import PlanAction, ActionType

# Liste des statistiques disponibles pour l'UI
STATS_AVAILABLE = {
    'moyenne': 'Moyenne (mean)',
    'somme': 'Somme totale (sum)',
    'minimum': 'Valeur minimale',
    'maximum': 'Valeur maximale',
    'mediane': 'Médiane (median)',
    'ecart_type': 'Écart-type (std)',
    'variance': 'Variance',
    'count': 'Nombre de valeurs',
    'percentile_25': '1er quartile (25%)',
    'percentile_75': '3ème quartile (75%)',
}


def calculate_stats(
    df: pd.DataFrame,
    columns: List[str],
    operation: str = 'moyenne',
    groupby: Optional[str] = None,
    **kwargs
) -> Union[pd.DataFrame, pd.Series, float]:
    """
    Calcule une statistique sur les colonnes spécifiées.
    
    Args:
        df: DataFrame source
        columns: Colonnes numériques à analyser
        operation: Type de statistique (voir STATS_AVAILABLE)
        groupby: Colonne de groupement optionnelle
    
    Returns:
        Résultat selon l'opération (DataFrame si groupby, valeur sinon)
    """
    
    # Vérification des colonnes
    available_cols = [c for c in columns if c in df.columns]
    numeric_cols = [c for c in available_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    if not numeric_cols:
        raise ValueError(f"Aucune colonne numérique valide parmi {columns}")
    
    # Mapping des opérations
    op_mapping = {
        'moyenne': 'mean',
        'somme': 'sum',
        'minimum': 'min',
        'maximum': 'max',
        'mediane': 'median',
        'ecart_type': 'std',
        'variance': 'var',
        'count': 'count',
        'percentile_25': lambda x: x.quantile(0.25),
        'percentile_75': lambda x: x.quantile(0.75),
    }
    
    agg_func = op_mapping.get(operation, 'mean')
    
    results = {}
    
    for col in numeric_cols:
        series = df[col]
        
        if groupby and groupby in df.columns:
            # Avec groupby - retourne DataFrame
            grouped = df.groupby(groupby)[col].agg(agg_func).reset_index()
            grouped.columns = [groupby, f"{operation}_{col}"]
            results[col] = grouped
        else:
            # Sans groupby - valeur scalaire
            if callable(agg_func):
                results[col] = agg_func(series)
            else:
                results[col] = getattr(series, agg_func)()
    
    # Formatage du résultat
    if groupby:
        if len(results) == 1:
            return list(results.values())[0]
        else:
            # Fusionner les résultats groupés
            result_df = list(results.values())[0]
            for col, df_res in list(results.items())[1:]:
                merge_col = f"{operation}_{col}"
                result_df = result_df.merge(
                    df_res, 
                    on=groupby, 
                    suffixes=('', f'_{col}')
                )
            return result_df
    else:
        if len(results) == 1:
            return list(results.values())[0]
        return pd.Series(results, name=operation)


def quick_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Statistiques rapides complètes sur une colonne.
    
    Returns:
        Dictionnaire avec toutes les stats de base
    """
    if column not in df.columns:
        raise ValueError(f"Colonne '{column}' introuvable")
    
    series = df[column]
    
    if not pd.api.types.is_numeric_dtype(series):
        # Stats pour catégoriel
        return {
            'type': 'categorical',
            'unique': series.nunique(),
            'mode': series.mode().iloc[0] if len(series.mode()) > 0 else None,
            'null_count': series.isnull().sum(),
            'null_pct': series.isnull().mean() * 100,
            'top_5': series.value_counts().head(5).to_dict()
        }
    
    # Stats pour numérique
    return {
        'type': 'numeric',
        'count': series.count(),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        '25%': series.quantile(0.25),
        'median': series.median(),
        '75%': series.quantile(0.75),
        'max': series.max(),
        'null_count': series.isnull().sum(),
        'null_pct': series.isnull().mean() * 100,
        'skewness': series.skew(),
        'kurtosis': series.kurtosis(),
    }
