"""
Module d'exécution des plans d'action.
Transforme les plans validés en opérations pandas concrètes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import io
import base64

from .parser import PlanAction, IntentionType, ActionType
from .validator import ValidationResult, ValidationLevel


class ExecutionStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"  # Réussi avec warnings
    FAILED = "failed"
    EMPTY_RESULT = "empty_result"


@dataclass
class ExecutionResult:
    """Résultat d'une exécution."""
    status: ExecutionStatus
    data: Optional[Any] = None  # DataFrame, Series, figure, ou valeur scalaire
    message: str = ""
    metadata: Dict[str, Any] = None  # Infos supplémentaires (stats, etc.)
    execution_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PlanExecutor:
    """
    Exécuteur de plans d'action.
    Transforme les intentions en code pandas/python exécutable.
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[Dict] = None):
        self.df = df.copy()  # On travaille sur une copie pour éviter les effets de bord
        self.original_df = df  # Référence au df original
        self.config = config or {}
        self.execution_log = []
        
        # Configuration par défaut
        self.default_figsize = self.config.get('figsize', (10, 6))
        self.style = self.config.get('style', 'seaborn-v0_8-darkgrid')
        plt.style.use(self.style)
        
    def execute(self, plan: PlanAction) -> ExecutionResult:
        """
        Exécute un plan d'action complet.
        
        Args:
            plan: Plan d'action validé
        
        Returns:
            ExecutionResult avec les données ou visualisation
        """
        import time
        start_time = time.time()
        
        try:
            # Appliquer les filtres d'abord
            filtered_df = self._apply_filters(self.df, plan.filters)
            
            # Exécuter selon l'intention
            if plan.intention == IntentionType.STATISTIQUE:
                result = self._execute_statistique(filtered_df, plan)
            elif plan.intention == IntentionType.VISUALISATION:
                result = self._execute_visualisation(filtered_df, plan)
            elif plan.intention == IntentionType.NETTOYAGE:
                result = self._execute_nettoyage(filtered_df, plan)
            elif plan.intention == IntentionType.EXPLORATION:
                result = self._execute_exploration(filtered_df, plan)
            else:
                result = ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    message="Intention non reconnue ou non supportée"
                )
            
            # Ajouter le temps d'exécution
            result.execution_time_ms = (time.time() - start_time) * 1000
            self.execution_log.append({
                'plan': plan,
                'result': result,
                'timestamp': pd.Timestamp.now()
            })
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message=f"Erreur d'exécution: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _apply_filters(self, df: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
        """Applique les filtres au dataframe."""
        if not filters:
            return df
        
        mask = pd.Series([True] * len(df), index=df.index)
        
        for filt in filters:
            col = filt['column']
            op = filt['operator']
            val = filt['value']
            
            if col not in df.columns:
                continue  # On ignore silencieusement, le validateur a déjà signalé l'erreur
            
            if op == '==':
                mask &= df[col] == val
            elif op == '!=':
                mask &= df[col] != val
            elif op == '>':
                mask &= df[col] > val
            elif op == '<':
                mask &= df[col] < val
            elif op == '>=':
                mask &= df[col] >= val
            elif op == '<=':
                mask &= df[col] <= val
            elif op == 'in':
                mask &= df[col].isin(val if isinstance(val, list) else [val])
            elif op == 'not in':
                mask &= ~df[col].isin(val if isinstance(val, list) else [val])
            elif op == 'contains':
                mask &= df[col].astype(str).str.contains(str(val), case=False, na=False)
            elif op == 'startswith':
                mask &= df[col].astype(str).str.startswith(str(val), na=False)
            elif op == 'endswith':
                mask &= df[col].astype(str).str.endswith(str(val), na=False)
        
        return df[mask].copy()
    
    def _execute_statistique(self, df: pd.DataFrame, plan: PlanAction) -> ExecutionResult:
        """Exécute les opérations statistiques."""
        cols = [c for c in plan.target_columns if c in df.columns]
        
        if not cols:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message="Aucune colonne valide pour l'analyse statistique"
            )
        
        # Déterminer si on fait un groupby
        groupby_cols = [c for c in plan.groupby_columns if c in df.columns]
        
        if groupby_cols:
            return self._execute_groupby_aggregation(df, plan, cols, groupby_cols)
        
        # Statistiques simples (sans groupby)
        result_data = {}
        
        for action in [plan.action]:  # On peut étendre pour supporter multi-actions
            if action == ActionType.MOYENNE:
                result_data = df[cols].mean().to_dict()
                message = f"Moyennes calculées sur {len(cols)} colonne(s)"
            
            elif action == ActionType.SOMME:
                result_data = df[cols].sum().to_dict()
                message = f"Sommes calculées sur {len(cols)} colonne(s)"
            
            elif action == ActionType.MEDIANE:
                result_data = df[cols].median().to_dict()
                message = f"Médianes calculées sur {len(cols)} colonne(s)"
            
            elif action == ActionType.ECART_TYPE:
                result_data = df[cols].std().to_dict()
                message = f"Écarts-types calculés sur {len(cols)} colonne(s)"
            
            elif action == ActionType.VARIANCE:
                result_data = df[cols].var().to_dict()
                message = f"Variances calculées sur {len(cols)} colonne(s)"
            
            elif action == ActionType.MINIMUM:
                result_data = df[cols].min().to_dict()
                message = f"Minimums trouvés sur {len(cols)} colonne(s)"
            
            elif action == ActionType.MAXIMUM:
                result_data = df[cols].max().to_dict()
                message = f"Maximums trouvés sur {len(cols)} colonne(s)"
            
            elif action == ActionType.COMPTE:
                result_data = df[cols].count().to_dict()
                message = f"Comptes effectués sur {len(cols)} colonne(s)"
            
            elif action == ActionType.DESCRIBE:
                result_data = df[cols].describe()
                message = f"Description statistique de {len(cols)} colonne(s)"
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    data=result_data,
                    message=message,
                    metadata={'type': 'dataframe', 'shape': result_data.shape}
                )
            
            elif action == ActionType.CORRELATION:
                if len(cols) >= 2:
                    result_data = df[cols].corr()
                    message = f"Matrice de corrélation ({len(cols)}x{len(cols)})"
                    return ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        data=result_data,
                        message=message,
                        metadata={'type': 'correlation_matrix', 'columns': cols}
                    )
                else:
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        message="La corrélation nécessite au moins 2 colonnes numériques"
                    )
            
            elif action == ActionType.REGRESSION:
                return self._execute_regression(df, plan, cols)
            
            else:
                # Action par défaut : describe
                result_data = df[cols].describe()
                message = f"Analyse descriptive (action par défaut)"
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    data=result_data,
                    message=message,
                    metadata={'type': 'dataframe', 'shape': result_data.shape}
                )
        
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            data=result_data,
            message=message,
            metadata={'type': 'dict', 'count': len(result_data)}
        )
    
    def _execute_groupby_aggregation(self, df: pd.DataFrame, plan: PlanAction, 
                                    target_cols: List[str], groupby_cols: List[str]) -> ExecutionResult:
        """Exécute les agrégations avec groupby."""
        
        # Mapper les actions aux fonctions d'agrégation
        agg_map = {
            ActionType.MOYENNE: 'mean',
            ActionType.SOMME: 'sum',
            ActionType.MEDIANE: 'median',
            ActionType.ECART_TYPE: 'std',
            ActionType.VARIANCE: 'var',
            ActionType.MINIMUM: 'min',
            ActionType.MAXIMUM: 'max',
            ActionType.COMPTE: 'count'
        }
        
        agg_func = agg_map.get(plan.action, 'mean')
        
        try:
            grouped = df.groupby(groupby_cols)[target_cols].agg(agg_func)
            
            # Si un seul groupe et une seule colonne, retourner une valeur scalaire
            if len(grouped) == 1 and len(target_cols) == 1:
                value = grouped.iloc[0]
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    data=value,
                    message=f"{agg_func} de {target_cols[0]} pour {groupby_cols[0]}={grouped.index[0]}: {value:.4f}",
                    metadata={'type': 'scalar', 'value': value}
                )
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                data=grouped,
                message=f"Agrégation '{agg_func}' par {', '.join(groupby_cols)}",
                metadata={
                    'type': 'grouped_dataframe',
                    'shape': grouped.shape,
                    'n_groups': len(grouped)
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message=f"Erreur lors du groupby: {str(e)}"
            )
    
    def _execute_regression(self, df: pd.DataFrame, plan: PlanAction, cols: List[str]) -> ExecutionResult:
        """Exécute une régression linéaire simple ou multiple."""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        
        try:
            # Préparer les données (gérer les NaN)
            data = df[cols].dropna()
            
            if len(cols) == 2:
                # Régression simple: X vs Y
                X = data[[cols[0]]].values
                y = data[cols[1]].values
                feature_names = [cols[0]]
            else:
                # Régression multiple: dernière colonne = Y, autres = X
                X = data[cols[:-1]].values
                y = data[cols[-1]].values
                feature_names = cols[:-1]
            
            if len(X) < 2:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    message="Pas assez de données pour la régression (minimum 2 points)"
                )
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            result = {
                'coefficients': dict(zip(feature_names, model.coef_)),
                'intercept': model.intercept_,
                'r_squared': r2,
                'rmse': rmse,
                'n_samples': len(X),
                'equation': f"{cols[-1] if len(cols) > 2 else cols[1]} = " + 
                           " + ".join([f"{coef:.4f}*{name}" for coef, name in zip(model.coef_, feature_names)]) +
                           f" + {model.intercept_:.4f}"
            }
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                data=result,
                message=f"Régression linéaire R²={r2:.4f}, RMSE={rmse:.4f}",
                metadata={'type': 'regression_results', 'model': model}
            )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message=f"Erreur régression: {str(e)}"
            )
    
    def _execute_visualisation(self, df: pd.DataFrame, plan: PlanAction) -> ExecutionResult:
        """Génère les visualisations."""
        cols = [c for c in plan.target_columns if c in df.columns]
        
        if not cols:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message="Aucune colonne valide pour la visualisation"
            )
        
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        try:
            if plan.action == ActionType.HISTOGRAMME:
                if len(cols) == 1:
                    df[cols[0]].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                    ax.set_title(f'Distribution de {cols[0]}')
                    ax.set_xlabel(cols[0])
                    ax.set_ylabel('Fréquence')
                else:
                    df[cols].hist(bins=30, ax=ax, figsize=(12, 8))
                    fig.suptitle('Distributions multiples')
            
            elif plan.action == ActionType.BAR_CHART:
                if plan.groupby_columns:
                    # Agrégation pour le bar chart
                    group_col = plan.groupby_columns[0]
                    if group_col in df.columns:
                        agg_data = df.groupby(group_col)[cols[0]].mean().sort_values(ascending=False).head(20)
                        agg_data.plot(kind='bar', ax=ax, color='steelblue')
                        ax.set_title(f'{cols[0]} moyen par {group_col}')
                    else:
                        value_counts = df[cols[0]].value_counts().head(20)
                        value_counts.plot(kind='bar', ax=ax, color='steelblue')
                        ax.set_title(f'Distribution de {cols[0]}')
                else:
                    value_counts = df[cols[0]].value_counts().head(20)
                    value_counts.plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_title(f'Distribution de {cols[0]}')
                ax.tick_params(axis='x', rotation=45)
            
            elif plan.action == ActionType.PIE_CHART:
                value_counts = df[cols[0]].value_counts().head(10)
                colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
                value_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors)
                ax.set_title(f'Répartition de {cols[0]}')
                ax.set_ylabel('')
            
            elif plan.action == ActionType.SCATTER_PLOT:
                if len(cols) >= 2:
                    # Gestion des gros datasets (échantillonnage)
                    plot_df = df[cols[:2]].dropna()
                    if len(plot_df) > 5000:
                        plot_df = plot_df.sample(5000, random_state=42)
                    
                    ax.scatter(plot_df[cols[0]], plot_df[cols[1]], alpha=0.5, s=20)
                    ax.set_xlabel(cols[0])
                    ax.set_ylabel(cols[1])
                    ax.set_title(f'{cols[1]} vs {cols[0]} (n={len(plot_df):,})')
                    
                    # Ajouter ligne de régression si pertinent
                    if len(plot_df) > 10:
                        z = np.polyfit(plot_df[cols[0]], plot_df[cols[1]], 1)
                        p = np.poly1d(z)
                        ax.plot(plot_df[cols[0]].sort_values(), 
                               p(plot_df[cols[0]].sort_values()), 
                               "r--", alpha=0.8, label='tendance')
                        ax.legend()
                else:
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        message="Scatter plot nécessite 2 colonnes numériques"
                    )
            
            elif plan.action == ActionType.LINE_CHART:
                if len(cols) >= 2 and pd.api.types.is_datetime64_any_dtype(df[cols[0]]):
                    # Time series
                    plot_df = df.set_index(cols[0])[cols[1]].sort_index()
                    plot_df.plot(ax=ax, linewidth=2)
                    ax.set_title(f'Évolution de {cols[1]}')
                    ax.set_ylabel(cols[1])
                else:
                    # Line chart simple
                    df[cols].plot(ax=ax, linewidth=2)
                    ax.set_title(f'Tendances: {", ".join(cols)}')
            
            elif plan.action == ActionType.BOXPLOT:
                if len(cols) == 1:
                    df[cols[0]].plot(kind='box', ax=ax)
                    ax.set_title(f'Boxplot de {cols[0]}')
                else:
                    df[cols].plot(kind='box', ax=ax)
                    ax.set_title('Boxplots comparatifs')
            
            elif plan.action == ActionType.HEATMAP:
                if len(cols) >= 2:
                    corr_matrix = df[cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                               square=True, ax=ax, fmt='.2f')
                    ax.set_title('Matrice de corrélation')
                else:
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        message="Heatmap nécessite au moins 2 colonnes numériques"
                    )
            
            else:
                # Visualisation par défaut
                df[cols[0]].hist(bins=30, ax=ax, edgecolor='black')
                ax.set_title(f'Distribution de {cols[0]} (défaut)')
            
            plt.tight_layout()
            
            # Convertir en base64 pour intégration web/API
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                data={'image_base64': img_base64, 'figure': fig},
                message=f"Visualisation '{plan.action.value}' générée",
                metadata={
                    'type': 'visualization',
                    'format': 'base64_png',
                    'columns': cols,
                    'n_points': len(df)
                }
            )
            
        except Exception as e:
            plt.close(fig)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message=f"Erreur visualisation: {str(e)}"
            )
    
    def _execute_nettoyage(self, df: pd.DataFrame, plan: PlanAction) -> ExecutionResult:
        """Exécute les opérations de nettoyage."""
        cols = [c for c in plan.target_columns if c in df.columns]
        modifications = []
        
        try:
            if plan.action == ActionType.SUPPRIMER_LIGNES_VIDES:
                before_len = len(df)
                df_clean = df.dropna(subset=cols if cols else None)
                after_len = len(df_clean)
                removed = before_len - after_len
                modifications.append(f"{removed:,} lignes supprimées")
                
            elif plan.action == ActionType.SUPPRIMER_DOUBLONS:
                before_len = len(df)
                subset_cols = cols if cols else None
                df_clean = df.drop_duplicates(subset=subset_cols)
                after_len = len(df_clean)
                removed = before_len - after_len
                modifications.append(f"{removed:,} doublons supprimés")
                
            elif plan.action == ActionType.REMPLIR_MANQUANT:
                df_clean = df.copy()
                for col in (cols if cols else df.columns):
                    if df_clean[col].dtype in ['int64', 'float64']:
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)
                        modifications.append(f"{col}: manquants remplis par médiane ({median_val:.2f})")
                    else:
                        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Inconnu"
                        df_clean[col] = df_clean[col].fillna(mode_val)
                        modifications.append(f"{col}: manquants remplis par mode ({mode_val})")
            
            elif plan.action == ActionType.DETECTER_OUTLIERS:
                outliers_info = {}
                for col in cols:
                    if df[col].dtype in ['int64', 'float64']:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                        outliers_info[col] = {
                            'count': len(outliers),
                            'percentage': len(outliers) / len(df) * 100,
                            'bounds': (lower_bound, upper_bound)
                        }
                
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    data=outliers_info,
                    message=f"Outliers détectés dans {len(outliers_info)} colonne(s)",
                    metadata={'type': 'outliers_report', 'details': outliers_info}
                )
            
            elif plan.action == ActionType.SUPPRIMER_COLONNES:
                df_clean = df.drop(columns=cols)
                modifications.append(f"{len(cols)} colonne(s) supprimée(s): {', '.join(cols)}")
            
            else:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    message=f"Action de nettoyage '{plan.action}' non reconnue"
                )
            
            # Mettre à jour le dataframe interne
            self.df = df_clean
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                data=df_clean,
                message="Nettoyage effectué: " + "; ".join(modifications),
                metadata={
                    'type': 'cleaned_dataframe',
                    'shape_before': self.original_df.shape,
                    'shape_after': df_clean.shape,
                    'modifications': modifications
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message=f"Erreur nettoyage: {str(e)}"
            )
    
    def _execute_exploration(self, df: pd.DataFrame, plan: PlanAction) -> ExecutionResult:
        """Exécute l'exploration de données."""
        
        if plan.action == ActionType.HEAD:
            n = 5  # Par défaut
            data = df.head(n)
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                data=data,
                message=f"Premières {len(data)} lignes",
                metadata={'type': 'dataframe_preview', 'shape': data.shape}
            )
        
        elif plan.action == ActionType.TAIL:
            n = 5
            data = df.tail(n)
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                data=data,
                message=f"Dernières {len(data)} lignes",
                metadata={'type': 'dataframe_preview', 'shape': data.shape}
            )
        
        elif plan.action == ActionType.INFO:
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            
            # Calculer des stats supplémentaires
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
            dtypes_count = df.dtypes.value_counts().to_dict()
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                data={
                    'info': info_str,
                    'memory_mb': memory_usage,
                    'dtypes': dtypes_count,
                    'columns_count': len(df.columns),
                    'rows_count': len(df)
                },
                message=f"Dataset: {len(df):,} lignes, {len(df.columns)} colonnes, {memory_usage:.2f} MB",
                metadata={'type': 'dataset_info'}
            )
        
        elif plan.action == ActionType.DESCRIBE:
            desc = df.describe(include='all').T
            desc['missing'] = df.isnull().sum()
            desc['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                data=desc,
                message=f"Description complète du dataset",
                metadata={'type': 'full_description', 'shape': desc.shape}
            )
        
        else:
            # Exploration par défaut: aperçu + info
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                data={
                    'head': df.head(),
                    'info': {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
                    }
                },
                message=f"Dataset: {len(df):,} lignes × {len(df.columns)} colonnes",
                metadata={'type': 'quick_explore'}
            )


# Fonctions utilitaires
def execute_plan(df: pd.DataFrame, plan: PlanAction, config: Optional[Dict] = None) -> ExecutionResult:
    """
    Exécute un plan d'action de manière simple.
    
    Args:
        df: DataFrame source
        plan: Plan d'action à exécuter
        config: Configuration optionnelle
    
    Returns:
        ExecutionResult
    """
    executor = PlanExecutor(df, config)
    return executor.execute(plan)


def quick_analyze(df: pd.DataFrame, query: str) -> ExecutionResult:
    """
    Analyse rapide: parse, valide et exécute en une fois.
    
    Args:
        df: DataFrame à analyser
        query: Requête en langage naturel
    
    Returns:
        ExecutionResult
    """
    from .parser import parse_query
    from .validator import validate_plan
    
    # Parser
    plan = parse_query(query, df.columns.tolist())
    
    # Valider
    is_valid, issues = validate_plan(df, plan)
    
    if not is_valid:
        error_msg = "\n".join([f"❌ {i.message}" for i in issues if i.level == ValidationLevel.ERROR])
        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            message=f"Validation échouée:\n{error_msg}"
        )
    
    # Exécuter
    return execute_plan(df, plan)
