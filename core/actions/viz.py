"""
Module de visualisations.
Génère des graphiques Plotly de manière sécurisée.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any, Union
import numpy as np

# Types de visualisations disponibles
VIZ_AVAILABLE = {
    'histogram': 'Histogramme de distribution',
    'bar': 'Diagramme en barres',
    'scatter': 'Nuage de points (XY)',
    'line': 'Graphique linéaire',
    'pie': 'Diagramme circulaire',
    'box': 'Boîte à moustaches',
    'heatmap': 'Carte de chaleur (corrélations)',
    'area': 'Graphique en aire',
    'violin': 'Violin plot',
}


def create_visualization(
    df: pd.DataFrame,
    viz_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Crée une visualisation Plotly de manière sécurisée.
    
    Args:
        df: DataFrame source
        viz_type: Type de visualisation (voir VIZ_AVAILABLE)
        x: Colonne pour l'axe X
        y: Colonne pour l'axe Y
        color: Colonne pour la couleur/séparation
        title: Titre personnalisé
        **kwargs: Options supplémentaires
    
    Returns:
        Figure Plotly prête à afficher
    """
    
    # Auto-détection si non spécifié
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if x is None and len(df.columns) > 0:
        x = categorical_cols[0] if categorical_cols and viz_type in ['bar', 'pie'] else numeric_cols[0] if numeric_cols else df.columns[0]
    
    if y is None and viz_type in ['scatter', 'line', 'bar']:
        y = numeric_cols[0] if numeric_cols else None
    
    # Génération du titre
    auto_title = title or generate_title(viz_type, x, y, color)
    
    # Dispatch selon le type
    if viz_type == 'histogram':
        fig = _make_histogram(df, x, color, auto_title, **kwargs)
    
    elif viz_type == 'bar':
        fig = _make_bar_chart(df, x, y, color, auto_title, **kwargs)
    
    elif viz_type == 'scatter':
        fig = _make_scatter(df, x, y, color, auto_title, **kwargs)
    
    elif viz_type == 'line':
        fig = _make_line_chart(df, x, y, color, auto_title, **kwargs)
    
    elif viz_type == 'pie':
        fig = _make_pie_chart(df, x, auto_title, **kwargs)
    
    elif viz_type == 'box':
        fig = _make_box_plot(df, x, y, color, auto_title, **kwargs)
    
    elif viz_type == 'heatmap':
        fig = _make_heatmap(df, auto_title, **kwargs)
    
    elif viz_type == 'area':
        fig = _make_area_chart(df, x, y, color, auto_title, **kwargs)
    
    elif viz_type == 'violin':
        fig = _make_violin_plot(df, x, y, color, auto_title, **kwargs)
    
    else:
        raise ValueError(f"Type de visualisation inconnu: {viz_type}")
    
    # Style commun
    fig.update_layout(
        template='plotly_white',
        height=kwargs.get('height', 500),
        width=kwargs.get('width', None),
        showlegend=kwargs.get('showlegend', True),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


def generate_title(viz_type: str, x: Optional[str], y: Optional[str], color: Optional[str]) -> str:
    """Génère un titre automatique."""
    parts = [VIZ_AVAILABLE.get(viz_type, viz_type)]
    
    if x:
        parts.append(f"de {x}")
    if y and y != x:
        parts.append(f"vs {y}")
    if color:
        parts.append(f"par {color}")
    
    return " ".join(parts)


def _make_histogram(df, x, color, title, **kwargs):
    """Crée un histogramme."""
    if pd.api.types.is_numeric_dtype(df[x]):
        return px.histogram(
            df, x=x, color=color,
            title=title,
            nbins=kwargs.get('nbins', 30),
            marginal=kwargs.get('marginal', 'box'),
            opacity=0.7
        )
    else:
        # Pour catégoriel, faire un bar chart
        counts = df[x].value_counts().head(kwargs.get('max_categories', 30))
        return px.bar(
            x=counts.index, y=counts.values,
            title=f"Fréquences de {x}",
            labels={'x': x, 'y': 'Count'}
        )


def _make_bar_chart(df, x, y, color, title, **kwargs):
    """Crée un diagramme en barres."""
    if y and y in df.columns:
        # y spécifié, agrégation si nécessaire
        if color:
            # Groupby pour éviter les doublons
            agg_df = df.groupby([x, color])[y].sum().reset_index()
        else:
            agg_df = df.groupby(x)[y].sum().reset_index()
        
        return px.bar(
            agg_df, x=x, y=y, color=color,
            title=title,
            barmode=kwargs.get('barmode', 'group')
        )
    else:
        # Compter les occurrences de x
        counts = df[x].value_counts().head(kwargs.get('max_bars', 20)).reset_index()
        counts.columns = [x, 'count']
        return px.bar(
            counts, x=x, y='count',
            title=title,
            color=x if len(counts) <= 10 else None
        )


def _make_scatter(df, x, y, color, title, **kwargs):
    """Crée un nuage de points."""
    # Échantillon pour la performance
    sample_size = kwargs.get('sample', min(len(df), 5000))
    df_sample = df.sample(min(len(df), sample_size)) if len(df) > sample_size else df
    
    return px.scatter(
        df_sample, x=x, y=y, color=color,
        title=title,
        opacity=kwargs.get('opacity', 0.6),
        size=kwargs.get('size'),
        hover_data=kwargs.get('hover_data', df.columns[:5].tolist())
    )


def _make_line_chart(df, x, y, color, title, **kwargs):
    """Crée un graphique linéaire."""
    # Trier si datetime
    if x and pd.api.types.is_datetime64_any_dtype(df[x]):
        df_sorted = df.sort_values(x)
    else:
        df_sorted = df
    
    # Agréger si trop de points
    if len(df_sorted) > 1000 and x:
        # Moyenne par groupe
        agg_df = df_sorted.groupby(x)[y].mean().reset_index()
        return px.line(agg_df, x=x, y=y, title=title + " (agrégé)")
    
    return px.line(
        df_sorted, x=x, y=y, color=color,
        title=title,
        markers=kwargs.get('markers', len(df) < 100)
    )


def _make_pie_chart(df, x, title, **kwargs):
    """Crée un diagramme circulaire."""
    counts = df[x].value_counts().head(kwargs.get('max_slices', 10))
    
    return px.pie(
        names=counts.index, values=counts.values,
        title=title,
        hole=kwargs.get('donut', False) * 0.4  # Donut si True
    )


def _make_box_plot(df, x, y, color, title, **kwargs):
    """Crée une boîte à moustaches."""
    if y and y in df.columns:
        # y est la valeur numérique, x optionnel pour grouper
        return px.box(
            df, x=x, y=y, color=color,
            title=title,
            points=kwargs.get('points', 'outliers')
        )
    else:
        # x est numérique, box simple
        return px.box(
            df, y=x,
            title=title,
            points='outliers'
        )


def _make_heatmap(df, title, **kwargs):
    """Crée une carte de chaleur de corrélations."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        raise ValueError("Heatmap nécessite 2+ colonnes numériques")
    
    # Calcul de corrélation
    corr_method = kwargs.get('method', 'pearson')
    corr_matrix = numeric_df.corr(method=corr_method)
    
    # Masquer la moitié si demandé
    if kwargs.get('mask_upper', False):
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix = corr_matrix.mask(mask)
    
    return px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        title=title,
        color_continuous_scale=kwargs.get('colorscale', 'RdBu_r'),
        zmid=0
    )


def _make_area_chart(df, x, y, color, title, **kwargs):
    """Crée un graphique en aire."""
    return px.area(
        df, x=x, y=y, color=color,
        title=title,
        line_shape=kwargs.get('line_shape', 'linear')
    )


def _make_violin_plot(df, x, y, color, title, **kwargs):
    """Crée un violin plot."""
    return px.violin(
        df, x=x, y=y, color=color,
        title=title,
        box=kwargs.get('show_box', True),
        points=kwargs.get('points', False)
    )
