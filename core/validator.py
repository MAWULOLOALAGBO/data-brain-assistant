"""
Module de validation des plans d'action.
Vérifie la cohérence logique, la faisabilité technique et la sécurité.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .parser import PlanAction, IntentionType, ActionType


class ValidationLevel(Enum):
    INFO = "info"           # Suggestion d'amélioration
    WARNING = "warning"     # Problème potentiel, mais exécutable
    ERROR = "error"         # Bloquant, ne peut pas exécuter


@dataclass
class ValidationResult:
    """Résultat d'une validation."""
    level: ValidationLevel
    code: str              # Code d'erreur unique
    message: str           # Message lisible
    suggestion: Optional[str] = None  # Comment corriger


class PlanValidator:
    """
    Validateur complet de plans d'action.
    Vérifie la logique métier et la faisabilité technique.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = list(df.columns)
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        self.issues = []
    
    def validate(self, plan: PlanAction) -> Tuple[bool, List[ValidationResult]]:
        """
        Valide un plan d'action complet.
        Retourne (est_valide, liste_des_problèmes).
        """
        self.issues = []
        
        # 1. Validation structurelle
        self._validate_structure(plan)
        
        # 2. Validation selon l'intention
        if plan.intention == IntentionType.STATISTIQUE:
            self._validate_statistique(plan)
        elif plan.intention == IntentionType.VISUALISATION:
            self._validate_visualisation(plan)
        elif plan.intention == IntentionType.NETTOYAGE:
            self._validate_nettoyage(plan)
        elif plan.intention == IntentionType.EXPLORATION:
            self._validate_exploration(plan)
        
        # 3. Validation du groupby
        self._validate_groupby(plan)
        
        # 4. Validation des filtres
        self._validate_filters(plan)
        
        # 5. Validation de cohérence globale
        self._validate_coherence(plan)
        
        # Déterminer si valide (pas d'erreurs bloquantes)
        is_valid = not any(issue.level == ValidationLevel.ERROR for issue in self.issues)
        
        return is_valid, self.issues
    
    def _validate_structure(self, plan: PlanAction):
        """Validation de base de la structure."""
        # Colonnes cibles existent ?
        for col in plan.target_columns:
            if col not in self.columns:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    code="COLONNE_INEXISTANTE",
                    message=f"La colonne '{col}' n'existe pas dans le dataset",
                    suggestion=f"Colonnes disponibles: {', '.join(self.columns[:5])}..."
                ))
        
        # Intention définie ?
        if plan.intention == IntentionType.INCONNU:
            self.issues.append(ValidationResult(
                level=ValidationLevel.ERROR,
                code="INTENTION_INCONNUE",
                message="Impossible de comprendre l'intention de la requête",
                suggestion="Essayez d'être plus précis: 'moyenne de X', 'histogramme de Y'..."
            ))
        
        # Action définie ?
        if plan.action == ActionType.INCONNU:
            self.issues.append(ValidationResult(
                level=ValidationLevel.WARNING,
                code="ACTION_INCONNUE",
                message="Action spécifique non reconnue, utilisation d'une action par défaut",
                suggestion=None
            ))
    
    def _validate_statistique(self, plan: PlanAction):
        """Validation pour les opérations statistiques."""
        if not plan.target_columns:
            return
        
        for col in plan.target_columns:
            # Vérifier que la colonne est numérique pour les stats
            needs_numeric = plan.action in [
                ActionType.MOYENNE, ActionType.SOMME, ActionType.ECART_TYPE,
                ActionType.VARIANCE, ActionType.MEDIANE, ActionType.MINIMUM,
                ActionType.MAXIMUM
            ]
            
            if needs_numeric and col not in self.numeric_columns:
                # Est-ce qu'on peut la convertir ?
                if col in self.categorical_columns:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        code="COLONNE_NON_NUMERIQUE",
                        message=f"La colonne '{col}' est catégorielle, pas numérique",
                        suggestion=f"Utilisez une colonne numérique comme: {', '.join(self.numeric_columns[:3])}"
                    ))
                elif col in self.datetime_columns:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        code="COLONNE_DATETIME",
                        message=f"La colonne '{col}' est une date. Conversion en timestamp pour le calcul",
                        suggestion="Pour des stats sur les dates, utilisez 'describe' ou extrayez l'année/mois"
                    ))
            
            # Vérifier les valeurs manquantes
            null_ratio = self.df[col].isnull().mean()
            if null_ratio > 0.5:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="TROP_DE_MANQUANTS",
                    message=f"La colonne '{col}' contient {null_ratio*100:.1f}% de valeurs manquantes",
                    suggestion="Envisagez de nettoyer les données d'abord avec 'valeurs manquantes'"
                ))
            
            # Vérifier la variance (stats inutiles si constant)
            if col in self.numeric_columns and self.df[col].nunique() == 1:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="COLONNE_CONSTANTE",
                    message=f"La colonne '{col}' a une seule valeur unique ({self.df[col].iloc[0]})",
                    suggestion="Les statistiques sur une colonne constante sont inutiles"
                ))

    def _validate_visualisation(self, plan: PlanAction):
        """Validation pour les visualisations."""
        if not plan.target_columns:
            return
        
        col = plan.target_columns[0]
        
        # Histogramme : besoin de données numériques ou beaucoup de catégories
        if plan.action == ActionType.HISTOGRAMME:
            if col in self.categorical_columns:
                n_unique = self.df[col].nunique()
                if n_unique > 50:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        code="TROP_DE_CATEGORIES_HIST",
                        message=f"'{col}' a {n_unique} catégories, l'histogramme sera illisible",
                        suggestion="Utilisez 'bar chart' ou filtrez les valeurs principales"
                    ))
        
        # Scatter plot : besoin de 2 colonnes numériques
        elif plan.action == ActionType.SCATTER_PLOT:
            if len(plan.target_columns) < 2:
                if len(self.numeric_columns) >= 2:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.INFO,
                        code="SCATTER_COLONNE_UNIQUE",
                        message=f"Une seule colonne spécifiée pour le scatter plot",
                        suggestion=f"Utilisez '{self.numeric_columns[0]} et {self.numeric_columns[1]}' ou laissez l'auto-détection"
                    ))
                else:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        code="PAS_ASSEZ_NUMERIQUES_SCATTER",
                        message=f"Scatter plot nécessite 2 colonnes numériques, vous en avez {len(self.numeric_columns)}",
                        suggestion=f"Colonnes numériques disponibles: {', '.join(self.numeric_columns) if self.numeric_columns else 'Aucune'}"
                    ))
        
        # Pie chart : limité en catégories
        elif plan.action == ActionType.PIE_CHART:
            if col in self.numeric_columns:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="PIE_CHART_NUMERIQUE",
                    message=f"'{col}' est numérique, le pie chart montre des proportions",
                    suggestion=f"Utilisez une colonne catégorielle comme: {', '.join(self.categorical_columns[:3]) if self.categorical_columns else 'N/A'}"
                ))
            elif col in self.categorical_columns:
                n_unique = self.df[col].nunique()
                if n_unique > 10:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        code="TROP_DE_SEGMENTS_PIE",
                        message=f"'{col}' a {n_unique} catégories, le pie chart sera illisible",
                        suggestion="Limitez aux 10 plus fréquentes ou utilisez un bar chart"
                    ))
        
        # Line chart : besoin d'une dimension temporelle ou ordinale
        elif plan.action == ActionType.LINE_CHART:
            if col in self.categorical_columns and self.df[col].nunique() > 100:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="LINE_CHAOTIQUE",
                    message=f"Trop de points pour un line chart avec '{col}' ({self.df[col].nunique()} valeurs)",
                    suggestion="Agrégez les données ou utilisez un échantillon"
                ))
        
        # Heatmap : besoin de corrélations
        elif plan.action == ActionType.HEATMAP:
            if len(self.numeric_columns) < 2:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    code="PAS_ASSEZ_NUMERIQUES_HEATMAP",
                    message=f"Heatmap de corrélation nécessite 2+ colonnes numériques",
                    suggestion=f"Vous n'avez que {len(self.numeric_columns)} colonne(s) numérique(s)"
                ))

    def _validate_nettoyage(self, plan: PlanAction):
        """Validation pour le nettoyage de données."""
        # Toujours faisable, mais on vérifie s'il y a quelque chose à nettoyer
        
        if plan.action == ActionType.DETECTER_MANQUANTS:
            total_null = self.df.isnull().sum().sum()
            if total_null == 0:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.INFO,
                    code="AUCUN_MANQUANT",
                    message="Aucune valeur manquante détectée dans le dataset",
                    suggestion="Pas d'action de nettoyage nécessaire"
                ))
        
        elif plan.action == ActionType.DETECTER_DOUBLONS:
            n_duplicates = self.df.duplicated().sum()
            if n_duplicates == 0:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.INFO,
                    code="AUCUN_DOUBLON",
                    message="Aucune ligne dupliquée détectée",
                    suggestion=None
                ))
            elif n_duplicates > len(self.df) * 0.5:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="TROP_DE_DOUBLONS",
                    message=f"{n_duplicates} doublons ({n_duplicates/len(self.df)*100:.1f}%) - vérifiez l'unicité des IDs",
                    suggestion="Les doublons massifs peuvent indiquer un problème de jointure"
                ))
        
        elif plan.action == ActionType.DETECTER_OUTLIERS:
            if not self.numeric_columns:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    code="PAS_DE_NUMERIQUES_OUTLIERS",
                    message="Détection d'outliers impossible sans colonnes numériques",
                    suggestion=None
                ))
            else:
                # Vérifier la taille pour IQR
                for col in self.numeric_columns[:3]:  # Vérifier les 3 premières
                    if len(self.df) < 10:
                        self.issues.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            code="DATASET_TROP_PETIT_OUTLIERS",
                            message=f"Dataset très petit ({len(self.df)} lignes), la détection d'outliers est peu fiable",
                            suggestion=None
                        ))
                        break

    def _validate_exploration(self, plan: PlanAction):
        """Validation pour l'exploration."""
        # Describe et dtypes toujours faisables
        
        if plan.action == ActionType.CORRELATION:
            n_numeric = len(self.numeric_columns)
            if n_numeric < 2:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    code="CORRELATION_IMPOSSIBLE",
                    message=f"Matrice de corrélation nécessite 2+ colonnes numériques, vous en avez {n_numeric}",
                    suggestion=None
                ))
            elif n_numeric > 20:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="TROP_DE_COLONNES_CORR",
                    message=f"{n_numeric} colonnes numériques - la matrice sera grande",
                    suggestion="Sélectionnez des colonnes spécifiques pour une meilleure lisibilité"
                ))

    def _validate_groupby(self, plan: PlanAction):
        """Validation du groupby."""
        if not plan.groupby_column:
            return
        
        groupby_col = plan.groupby_column
        
        # La colonne existe ?
        if groupby_col not in self.columns:
            self.issues.append(ValidationResult(
                level=ValidationLevel.ERROR,
                code="GROUPBY_COLONNE_INEXISTANTE",
                message=f"Colonne de groupby '{groupby_col}' introuvable",
                suggestion=None
            ))
            return
        
        # La colonne est-elle catégorielle ?
        if groupby_col in self.numeric_columns:
            n_unique = self.df[groupby_col].nunique()
            if n_unique > 1000:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="GROUPBY_NUMERIQUE_DENSE",
                    message=f"'{groupby_col}' est numérique avec {n_unique} valeurs uniques",
                    suggestion="Envisagez de discrétiser en bins ou utilisez une colonne catégorielle"
                ))
        
        # Trop de groupes ?
        n_groups = self.df[groupby_col].nunique()
        if n_groups > 1000:
            self.issues.append(ValidationResult(
                level=ValidationLevel.WARNING,
                code="TROP_DE_GROUPES",
                message=f"{n_groups} groupes uniques - le résultat sera volumineux",
                suggestion="Filtrez les groupes principaux ou agrégez"
            ))
        elif n_groups == 1:
            self.issues.append(ValidationResult(
                level=ValidationLevel.INFO,
                code="UN_SEUL_GROUPE",
                message=f"Une seule valeur dans '{groupby_col}', le groupby est inutile",
                suggestion="Supprimez 'par {groupby_col}' de votre requête"
            ))
        
        # Y a-t-il des valeurs manquantes dans le groupby ?
        null_ratio = self.df[groupby_col].isnull().mean()
        if null_ratio > 0:
            self.issues.append(ValidationResult(
                level=ValidationLevel.WARNING,
                code="MANQUANTS_DANS_GROUPBY",
                message=f"{null_ratio*100:.1f}% de valeurs manquantes dans '{groupby_col}'",
                suggestion="Ces lignes seront exclues du groupby"
            ))

    def _validate_filters(self, plan: PlanAction):
        """Validation des conditions de filtrage."""
        for i, filter_cond in enumerate(plan.filter_conditions):
            # Vérifier que les valeurs ont du sens
            if filter_cond.get('operator') in ['>', '<', '>=', '<=']:
                value = filter_cond.get('value')
                try:
                    float(value)
                except (ValueError, TypeError):
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        code="FILTRE_VALEUR_NON_NUMERIQUE",
                        message=f"Filtre #{i+1}: '{value}' n'est pas un nombre pour une comparaison",
                        suggestion=None
                    ))

    def _validate_coherence(self, plan: PlanAction):
        """Validation de cohérence globale."""
        # Vérifier que le plan a un sens dans son ensemble
        
        # 1. Groupby sans colonnes cibles numériques pour les stats
        if plan.groupby_column and plan.intention == IntentionType.STATISTIQUE:
            if not plan.target_columns:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    code="GROUPBY_SANS_COLONNE_CIBLE",
                    message="Groupby spécifié mais aucune colonne à agréger",
                    suggestion="Précisez quelle colonne numérique agréger, ex: 'moyenne de Age par Gender'"
                ))
        
        # 2. Visualisation sans données suffisantes
        if plan.intention == IntentionType.VISUALISATION and len(self.df) == 0:
            self.issues.append(ValidationResult(
                level=ValidationLevel.ERROR,
                code="DATASET_VIDE",
                message="Impossible de visualiser un dataset vide",
                suggestion=None
            ))
        
        # 3. Trop de colonnes cibles
        if len(plan.target_columns) > 5:
            self.issues.append(ValidationResult(
                level=ValidationLevel.WARNING,
                code="TROP_DE_COLONNES_CIBLES",
                message=f"{len(plan.target_columns)} colonnes cibles - limitez à 2-3 pour la lisibilité",
                suggestion=f"Concentrez-vous sur: {', '.join(plan.target_columns[:3])}"
            ))


def validate_plan(plan: PlanAction, df: pd.DataFrame) -> Tuple[bool, List[ValidationResult]]:
    """
    Fonction utilitaire pour valider un plan.
    
    Returns:
        (is_valid, list_of_issues)
    """
    validator = PlanValidator(df)
    return validator.validate(plan)
