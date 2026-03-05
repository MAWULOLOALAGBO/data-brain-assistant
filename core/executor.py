"""
Module d'exécution des plans d'action.
SANS exec() - Toutes les actions sont des fonctions Python explicites.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, Tuple, Optional
import streamlit as st

from .parser import PlanAction, IntentionType, ActionType
from .validator import validate_plan, ValidationLevel


class ExecutionError(Exception):
    """Erreur d'exécution d'une action."""
    pass


class ActionExecutor:
    """
    Exécuteur d'actions. Chaque action est une méthode explicite, pas de code dynamique.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def execute(self, plan: PlanAction) -> Tuple[Any, Optional[go.Figure], Dict]:
        """
        Exécute un plan d'action validé.
        
        Returns:
            (result_data, plotly_figure_or_None, metadata_dict)
        """
        # Validation pré-exécution
        is_valid, issues = validate_plan(plan, self.df)
        
        # Bloquer si erreurs critiques
        critical_errors = [i for i in issues if i.level == ValidationLevel.ERROR]
        if critical_errors:
            error_msg = "\n".join([f"• {e.message}" for e in critical_errors])
            raise ExecutionError(f"Plan invalide:\n{error_msg}")
        
        # Dispatch selon l'intention
        if plan.intention == IntentionType.STATISTIQUE:
            return self._execute_statistique(plan)
        elif plan.intention == IntentionType.VISUALISATION:
            return self._execute_visualisation(plan)
        elif plan.intention == IntentionType.NETTOYAGE:
            return self._execute_nettoyage(plan)
        elif plan.intention == IntentionType.EXPLORATION:
            return self._execute_exploration(plan)
        else:
            raise ExecutionError(f"Intention non implémentée: {plan.intention}")
    
    def _get_column_safely(self, col_name: str) -> pd.Series:
        """Récupère une colonne avec vérification."""
        if col_name not in self.df.columns:
            raise ExecutionError(f"Colonne '{col_name}' introuvable")
        return self.df[col_name]
    
    def _apply_groupby(self, data: pd.DataFrame, plan: PlanAction) -> pd.DataFrame:
        """Applique le groupby si spécifié."""
        if not plan.groupby_column:
            return data
        
        groupby_col = plan.groupby_column
        if groupby_col not in data.columns:
            raise ExecutionError(f"Colonne de groupby '{groupby_col}' introuvable")
        
        return data
    
    def _execute_statistique(self, plan: PlanAction) -> Tuple[Any, None, Dict]:
        """Exécute une action statistique."""
        action = plan.action
        target_cols = plan.target_columns or [self.numeric_columns[0] if self.numeric_columns else self.df.columns[0]]
        groupby_col = plan.groupby_column
        
        results = {}
        metadata = {'action': action.value, 'columns': target_cols, 'groupby': groupby_col}
        
        for col in target_cols:
            series = self._get_column_safely(col)
            
            # Conversion si nécessaire
            if pd.api.types.is_datetime64_any_dtype(series):
                series = series.astype('int64')  # Timestamp pour calculs
            
            if not pd.api.types.is_numeric_dtype(series):
                continue  # Skip non-numériques (déjà warning dans validator)
            
            if groupby_col:
                # Groupby avec aggregation multiple
                grouped = self.df.groupby(groupby_col)[col].agg([
                    ('moyenne', 'mean'),
                    ('somme', 'sum'),
                    ('minimum', 'min'),
                    ('maximum', 'max'),
                    ('mediane', 'median'),
                    ('ecart_type', 'std'),
                    ('count', 'count')
                ]).reset_index()
                
                # Sélectionner selon l'action demandée
                if action == ActionType.MOYENNE:
                    result_df = grouped[[groupby_col, 'moyenne']].rename(columns={'moyenne': col})
                elif action == ActionType.SOMME:
                    result_df = grouped[[groupby_col, 'somme']].rename(columns={'somme': col})
                elif action == ActionType.MINIMUM:
                    result_df = grouped[[groupby_col, 'minimum']].rename(columns={'minimum': col})
                elif action == ActionType.MAXIMUM:
                    result_df = grouped[[groupby_col, 'maximum']].rename(columns={'maximum': col})
                elif action == ActionType.MEDIANE:
                    result_df = grouped[[groupby_col, 'mediane']].rename(columns={'mediane': col})
                elif action == ActionType.ECART_TYPE:
                    result_df = grouped[[groupby_col, 'ecart_type']].rename(columns={'ecart_type': col})
                else:
                    result_df = grouped  # Toutes les stats
                
                results[col] = result_df
            else:
                # Sans groupby - résultat scalaire ou série
                if action == ActionType.MOYENNE:
                    results[col] = series.mean()
                elif action == ActionType.SOMME:
                    results[col] = series.sum()
                elif action == ActionType.MINIMUM:
                    results[col] = series.min()
                elif action == ActionType.MAXIMUM:
                    results[col] = series.max()
                elif action == ActionType.MEDIANE:
                    results[col] = series.median()
                elif action == ActionType.ECART_TYPE:
                    results[col] = series.std()
                elif action == ActionType.VARIANCE:
                    results[col] = series.var()
                elif action == ActionType.COUNT:
                    results[col] = series.count()
                else:
                    # Toutes les stats par défaut
                    results[col] = series.agg(['mean', 'std', 'min', 'max', 'median', 'count'])
        
        # Formatage du résultat final
        if groupby_col:
            # Retourner DataFrame groupé
            if len(results) == 1:
                final_result = list(results.values())[0]
            else:
                # Fusionner plusieurs colonnes
                final_result = list(results.values())[0]
                for col, df_result in list(results.items())[1:]:
                    final_result = final_result.merge(df_result, on=groupby_col, suffixes=('', f'_{col}'))
        else:
            # Retourner série ou valeur unique
            if len(results) == 1:
                final_result = list(results.values())[0]
            else:
                final_result = pd.Series(results)
        
        return final_result, None, metadata
    
    def _execute_visualisation(self, plan: PlanAction) -> Tuple[str, go.Figure, Dict]:
        """Exécute une visualisation."""
        action = plan.action
        target_cols = plan.target_columns or [self.numeric_columns[0] if self.numeric_columns else self.df.columns[0]]
        groupby_col = plan.groupby_column
        
        col = target_cols[0]
        metadata = {'action': action.value, 'column': col, 'groupby': groupby_col}
        
        # Génération d'une clé unique pour Streamlit
        import time
        fig_key = f"fig_{int(time.time() * 1000) % 1000000}"
        
        if action == ActionType.HISTOGRAMME:
            if col in self.numeric_columns:
                fig = px.histogram(self.df, x=col, title=f"Distribution de {col}", nbins=30)
            else:
                # Pour catégoriel, compter les valeurs
                value_counts = self.df[col].value_counts().head(30)
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f"Fréquences de {col}", labels={'x': col, 'y': 'Count'})
        
        elif action == ActionType.BAR_CHART:
            if groupby_col and target_cols:
                # Agrégation par groupe
                agg_col = target_cols[0] if target_cols[0] in self.numeric_columns else self.numeric_columns[0]
                agg_df = self.df.groupby(groupby_col)[agg_col].sum().reset_index().sort_values(agg_col, ascending=False).head(20)
                fig = px.bar(agg_df, x=groupby_col, y=agg_col, 
                           title=f"{agg_col} par {groupby_col}")
            else:
                # Top valeurs catégorielles
                if col in self.categorical_columns:
                    value_counts = self.df[col].value_counts().head(15).reset_index()
                    value_counts.columns = [col, 'count']
                    fig = px.bar(value_counts, x=col, y='count', title=f"Top 15 {col}")
                else:
                    # Valeurs numériques
                    fig = px.bar(self.df.head(50), x=self.df.index[:50], y=col, title=f"Valeurs de {col}")
        
        elif action == ActionType.SCATTER_PLOT:
            if len(target_cols) >= 2 and target_cols[0] in self.numeric_columns and target_cols[1] in self.numeric_columns:
                x_col, y_col = target_cols[0], target_cols[1]
            elif len(self.numeric_columns) >= 2:
                x_col, y_col = self.numeric_columns[0], self.numeric_columns[1]
            else:
                raise ExecutionError("Scatter plot nécessite 2 colonnes numériques")
            
            color_arg = {groupby_col: self.df[groupby_col]} if groupby_col else None
            fig = px.scatter(self.df, x=x_col, y=y_col, color=groupby_col,
                           title=f"{x_col} vs {y_col}", opacity=0.6)
        
        elif action == ActionType.LINE_CHART:
            if groupby_col:
                agg_col = target_cols[0] if target_cols[0] in self.numeric_columns else self.numeric_columns[0]
                agg_df = self.df.groupby(groupby_col)[agg_col].mean().reset_index().sort_values(groupby_col)
                fig = px.line(agg_df, x=groupby_col, y=agg_col, title=f"{agg_col} par {groupby_col}")
            else:
                y_col = col if col in self.numeric_columns else self.numeric_columns[0]
                fig = px.line(self.df, y=y_col, title=f"Évolution de {y_col}")
        
        elif action == ActionType.PIE_CHART:
            if col in self.categorical_columns:
                value_counts = self.df[col].value_counts().head(10)
                fig = px.pie(names=value_counts.index, values=value_counts.values, title=f"Répartition {col}")
            else:
                raise ExecutionError(f"Pie chart nécessite une colonne catégorielle, '{col}' est numérique")
        
        elif action == ActionType.BOX_PLOT:
            if col in self.numeric_columns:
                fig = px.box(self.df, y=col, title=f"Box plot de {col}")
            else:
                raise ExecutionError(f"Box plot nécessite une colonne numérique")
        
        elif action == ActionType.HEATMAP:
            if len(self.numeric_columns) >= 2:
                corr_matrix = self.df[self.numeric_columns].corr()
                fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto', 
                              title='Matrice de corrélation', color_continuous_scale='RdBu_r')
            else:
                raise ExecutionError("Heatmap nécessite 2+ colonnes numériques")
        
        else:
            raise ExecutionError(f"Visualisation non implémentée: {action}")
        
        # Configuration commune
        fig.update_layout(
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return f"Graphique généré: {action.value}", fig, metadata
    
    def _execute_nettoyage(self, plan: PlanAction) -> Tuple[Any, None, Dict]:
        """Exécute une action de nettoyage."""
        action = plan.action
        metadata = {'action': action.value}
        
        if action == ActionType.DETECTER_MANQUANTS:
            null_counts = self.df.isnull().sum().sort_values(ascending=False)
            null_counts = null_counts[null_counts > 0]
            
            if len(null_counts) == 0:
                result = "✅ Aucune valeur manquante détectée"
            else:
                null_pct = (null_counts / len(self.df) * 100).round(2)
                result_df = pd.DataFrame({
                    'colonne': null_counts.index,
                    'manquants': null_counts.values,
                    'pourcentage': null_pct.values
                })
                result = result_df
            
            return result, None, metadata
        
        elif action == ActionType.DETECTER_DOUBLONS:
            n_duplicates = self.df.duplicated().sum()
            if n_duplicates == 0:
                result = "✅ Aucune ligne dupliquée"
            else:
                dup_pct = n_duplicates / len(self.df) * 100
                result = f"⚠️ {n_duplicates} lignes dupliquées ({dup_pct:.2f}%)"
                # Montrer un exemple
                dups = self.df[self.df.duplicated(keep=False)].head(10)
                metadata['exemple_duplicatas'] = dups
            
            return result, None, metadata
        
        elif action == ActionType.DETECTER_OUTLIERS:
            if not self.numeric_columns:
                raise ExecutionError("Pas de colonnes numériques pour détecter les outliers")
            
            col = plan.target_columns[0] if plan.target_columns else self.numeric_columns[0]
            series = self._get_column_safely(col)
            
            # Méthode IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(series < lower_bound) | (series > upper_bound)]
            
            result = {
                'colonne': col,
                'bornes': f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                'nombre_outliers': len(outliers),
                'pourcentage': len(outliers) / len(self.df) * 100,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }
            
            return result, None, metadata
        
        else:
            raise ExecutionError(f"Action de nettoyage non implémentée: {action}")
    
    def _execute_exploration(self, plan: PlanAction) -> Tuple[Any, None, Dict]:
        """Exécute une action d'exploration."""
        action = plan.action
        metadata = {'action': action.value}
        
        if action == ActionType.DESCRIBE:
            result = self.df.describe(include='all').transpose()
            return result, None, metadata
        
        elif action == ActionType.DTYPES:
            info_df = pd.DataFrame({
                'colonne': self.df.columns,
                'type': self.df.dtypes.astype(str),
                'non_null': self.df.count(),
                'null': self.df.isnull().sum(),
                'unique': self.df.nunique(),
                'memory_mb': [self.df[col].memory_usage(deep=True) / 1024**2 for col in self.df.columns]
            })
            return info_df, None, metadata
        
        elif action == ActionType.CORRELATION:
            if len(self.numeric_columns) < 2:
                raise ExecutionError("Corrélation nécessite 2+ colonnes numériques")
            
            corr_matrix = self.df[self.numeric_columns].corr()
            return corr_matrix, None, metadata
        
        elif action == ActionType.HEAD:
            return self.df.head(10), None, metadata
        
        elif action == ActionType.TAIL:
            return self.df.tail(10), None, metadata
        
        else:
            raise ExecutionError(f"Action d'exploration non implémentée: {action}")


def execute_action(plan: PlanAction, df: pd.DataFrame) -> Tuple[Any, Optional[go.Figure], Dict]:
    """
    Fonction utilitaire pour exécuter un plan.
    
    Returns:
        (result_data, plotly_figure_or_None, metadata)
    """
    executor = ActionExecutor(df)
    return executor.execute(plan)
