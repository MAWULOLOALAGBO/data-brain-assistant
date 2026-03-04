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
                ActionType.MAXIMUM, ActionType.CORRELATION, ActionType.REGRESSION
            ]
            
            if needs_numeric and col not in self.numeric_columns:
                # Est-ce qu'on peut la convertir ?
                if col in self.categorical_columns:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        code="COLONNE_NON_NUMERIQUE",
                        message=f"La colonne '{col}' n'est pas numérique (type: {self.df[col].dtype})",
                        suggestion=f"Convertissez la colonne en numérique ou choisissez une colonne numérique parmi: {', '.join(self.numeric_columns[:5])}"
                    ))
                else:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        code="TYPE_INCOMPATIBLE",
                        message=f"Type de données incompatible pour '{col}' ({self.df[col].dtype})",
                        suggestion="Vérifiez le type de la colonne ou nettoyez les données"
                    ))
            
            # Vérifier les valeurs manquantes excessives
            missing_ratio = self.df[col].isna().mean()
            if missing_ratio > 0.5:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="TROP_DE_VALEURS_MANQUANTES",
                    message=f"La colonne '{col}' contient {missing_ratio:.1%} de valeurs manquantes",
                    suggestion="Envisagez de nettoyer la colonne avant l'analyse statistique"
                ))
            
            # Vérifier la variance pour éviter division par zéro
            if plan.action in [ActionType.ECART_TYPE, ActionType.VARIANCE, ActionType.CORRELATION]:
                if col in self.numeric_columns and self.df[col].var() == 0:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        code="VARIANCE_NULLE",
                        message=f"La colonne '{col}' a une variance nulle (toutes les valeurs identiques)",
                        suggestion="Impossible de calculer l'écart-type ou la corrélation sur une constante"
                    ))
        
        # Validation spécifique pour la corrélation (besoin de 2+ colonnes)
        if plan.action == ActionType.CORRELATION and len(plan.target_columns) < 2:
            self.issues.append(ValidationResult(
                level=ValidationLevel.ERROR,
                code="CORRELATION_COLONNES_INSUFFISANTES",
                message="La corrélation nécessite au moins 2 colonnes numériques",
                suggestion=f"Ajoutez une colonne parmi: {', '.join(self.numeric_columns[:5])}"
            ))
        
        # Validation pour la régression
        if plan.action == ActionType.REGRESSION:
            if len(plan.target_columns) < 2:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    code="REGRESSION_COLONNES_INSUFFISANTES",
                    message="La régression nécessite au moins 2 colonnes (X et Y)",
                    suggestion="Spécifiez les variables indépendantes et dépendantes"
                ))
    
    def _validate_visualisation(self, plan: PlanAction):
        """Validation pour les visualisations."""
        if not plan.target_columns:
            self.issues.append(ValidationResult(
                level=ValidationLevel.WARNING,
                code="AUCUNE_COLONNE_VISUALISATION",
                message="Aucune colonne spécifiée pour la visualisation",
                suggestion="Précisez quelles colonnes visualiser"
            ))
            return
        
        for col in plan.target_columns:
            # Vérifier le nombre de catégories uniques pour les graphiques catégoriels
            if plan.action in [ActionType.HISTOGRAMME, ActionType.BAR_CHART, ActionType.PIE_CHART]:
                n_unique = self.df[col].nunique(dropna=True)
                
                if n_unique > 50:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        code="TROP_DE_CATEGORIES",
                        message=f"La colonne '{col}' a {n_unique} catégories uniques",
                        suggestion="Envisagez de regrouper les catégories ou utilisez un autre type de graphique"
                    ))
                
                if n_unique == 1:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        code="UNE_SEULE_CATEGORIE",
                        message=f"La colonne '{col}' n'a qu'une seule valeur unique",
                        suggestion="Le graphique sera peu informatif avec une seule catégorie"
                    ))
            
            # Vérifier pour les séries temporelles
            if plan.action == ActionType.LINE_CHART:
                if col not in self.datetime_columns and not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    # Vérifier si on peut parser comme date
                    try:
                        pd.to_datetime(self.df[col].dropna().iloc[:5])
                        self.issues.append(ValidationResult(
                            level=ValidationLevel.INFO,
                            code="CONVERSION_DATE_POSSIBLE",
                            message=f"La colonne '{col}' pourrait être convertie en datetime",
                            suggestion="La conversion automatique sera tentée pour le line chart"
                        ))
                    except (ValueError, TypeError):
                        self.issues.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            code="PAS_DE_DIMENSION_TEMPS",
                            message=f"La colonne '{col}' ne semble pas être une série temporelle",
                            suggestion="Utilisez une colonne de type date pour un line chart pertinent"
                        ))
            
            # Vérifier la taille pour les scatter plots
            if plan.action == ActionType.SCATTER_PLOT:
                if len(self.df) > 10000:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        code="DATASET_TRES_GRAND",
                        message=f"Dataset très grand ({len(self.df):,} lignes) pour un scatter plot",
                        suggestion="Envisagez d'échantillonner les données ou utiliser un hexbin plot"
                    ))
                
                if len(plan.target_columns) < 2:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        code="SCATTER_BESOIN_2_COLONNES",
                        message="Le scatter plot nécessite 2 colonnes numériques",
                        suggestion="Spécifiez X et Y pour le scatter plot"
                    ))
    
    def _validate_nettoyage(self, plan: PlanAction):
        """Validation pour les opérations de nettoyage."""
        if not plan.target_columns and plan.action != ActionType.NETTOYAGE_GLOBAL:
            self.issues.append(ValidationResult(
                level=ValidationLevel.ERROR,
                code="COLONNES_NETTOYAGE_REQUISES",
                message="Spécifiez les colonnes à nettoyer",
                suggestion="Ajoutez les colonnes cibles ou utilisez 'nettoyer tout'"
            ))
            return
        
        for col in plan.target_columns:
            # Vérifier les valeurs manquantes avant suppression
            if plan.action == ActionType.SUPPRIMER_LIGNES_VIDES:
                missing_count = self.df[col].isna().sum()
                if missing_count == 0:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.INFO,
                        code="AUCUNE_VALEUR_MANQUANTE",
                        message=f"La colonne '{col}' n'a pas de valeurs manquantes",
                        suggestion="Aucune action nécessaire sur cette colonne"
                    ))
                elif missing_count / len(self.df) > 0.3:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        code="RISQUE_PERTE_DONNEES",
                        message=f"Supprimer les lignes vides de '{col}' éliminera {missing_count:,} lignes ({missing_count/len(self.df):.1%})",
                        suggestion="Envisagez plutôt l'imputation ou la suppression de la colonne"
                    ))
            
            # Vérifier les doublons
            if plan.action == ActionType.SUPPRIMER_DOUBLONS:
                n_duplicates = self.df.duplicated(subset=[col]).sum()
                if n_duplicates == 0:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.INFO,
                        code="AUCUN_DOUBLON",
                        message=f"Aucun doublon trouvé dans '{col}'",
                        suggestion="Pas d'action nécessaire"
                    ))
            
            # Vérifier les outliers
            if plan.action == ActionType.DETECTER_OUTLIERS:
                if col not in self.numeric_columns:
                    self.issues.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        code="OUTLIERS_NON_NUMERIQUE",
                        message=f"La détection d'outliers nécessite une colonne numérique, '{col}' est {self.df[col].dtype}",
                        suggestion=f"Choisissez une colonne numérique parmi: {', '.join(self.numeric_columns[:5])}"
                    ))
    
    def _validate_exploration(self, plan: PlanAction):
        """Validation pour l'exploration de données."""
        # Généralement peu de contraintes, mais on peut ajouter des suggestions
        
        if plan.action == ActionType.DESCRIBE:
            if len(plan.target_columns) > 10:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.INFO,
                    code="MANY_COLUMNS_DESCRIBE",
                    message=f"Description de {len(plan.target_columns)} colonnes demandée",
                    suggestion="Envisagez de filtrer les colonnes pour une analyse plus ciblée"
                ))
        
        if plan.action == ActionType.HEAD or plan.action == ActionType.TAIL:
            # Pas de validation critique, mais on peut suggérer des filtres
            pass
    
    def _validate_groupby(self, plan: PlanAction):
        """Validation des opérations de groupby."""
        if not plan.groupby_columns:
            return
        
        for col in plan.groupby_columns:
            if col not in self.columns:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    code="GROUPBY_COLONNE_INEXISTANTE",
                    message=f"La colonne de groupby '{col}' n'existe pas",
                    suggestion=f"Colonnes disponibles: {', '.join(self.columns[:5])}"
                ))
                continue
            
            # Vérifier le nombre de groupes
            n_groups = self.df[col].nunique(dropna=True)
            if n_groups > 100:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="TROP_DE_GROUPES",
                    message=f"La colonne '{col}' crée {n_groups} groupes",
                    suggestion="Envisagez de regrouper les valeurs rares ou d'utiliser une autre colonne"
                ))
            
            if n_groups == 1:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="UN_SEUL_GROUPE",
                    message=f"La colonne '{col}' ne crée qu'un seul groupe",
                    suggestion="Le groupby n'aura aucun effet avec une seule catégorie"
                ))
            
            # Vérifier les valeurs manquantes dans le groupby
            missing_in_group = self.df[col].isna().sum()
            if missing_in_group > 0:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="VALEURS_MANQUANTES_GROUPBY",
                    message=f"{missing_in_group:,} valeurs manquantes dans '{col}' seront exclues du groupby",
                    suggestion="Envisagez de remplir les valeurs manquantes avant le groupby"
                ))
        
        # Vérifier la cohérence groupby + agrégation
        if plan.groupby_columns and plan.intention == IntentionType.STATISTIQUE:
            if not plan.target_columns:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    code="GROUPBY_SANS_AGGREGATION",
                    message="Groupby spécifié mais aucune colonne à agréger",
                    suggestion="Précisez quelles colonnes agréger (ex: 'moyenne de X par Y')"
                ))
    
    def _validate_filters(self, plan: PlanAction):
        """Validation des filtres."""
        if not plan.filters:
            return
        
        for i, filt in enumerate(plan.filters):
            col = filt.get('column')
            op = filt.get('operator')
            val = filt.get('value')
            
            if col not in self.columns:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    code=f"FILTRE_COLONNE_INEXISTANTE_{i}",
                    message=f"Le filtre #{i+1} référence une colonne inexistante: '{col}'",
                    suggestion=f"Colonnes disponibles: {', '.join(self.columns[:5])}"
                ))
                continue
            
            # Vérifier l'opérateur
            valid_operators = ['==', '!=', '>', '<', '>=', '<=', 'in', 'not in', 'contains', 'startswith', 'endswith']
            if op not in valid_operators:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    code=f"OPERATEUR_INVALIDE_{i}",
                    message=f"Opérateur '{op}' non reconnu dans le filtre #{i+1}",
                    suggestion=f"Opérateurs valides: {', '.join(valid_operators)}"
                ))
            
            # Vérifier la cohérence type/valeur
            if col in self.numeric_columns:
                try:
                    float(val)
                except (ValueError, TypeError):
                    if op not in ['in', 'not in']:  # Pour in/not in, on accepte les listes
                        self.issues.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            code=f"TYPE_FILTRE_MISMATCH_{i}",
                            message=f"La valeur '{val}' ne semble pas numérique pour la colonne '{col}'",
                            suggestion=f"Utilisez une valeur numérique ou convertissez la colonne"
                        ))
            
            # Vérifier si le filtre va tout éliminer
            if col in self.columns:
                try:
                    mask = self._apply_filter_mask(col, op, val)
                    n_remaining = mask.sum()
                    if n_remaining == 0:
                        self.issues.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            code=f"FILTRE_TROP_RESTRICTIF_{i}",
                            message=f"Le filtre #{i+1} ({col} {op} {val}) élimine toutes les lignes",
                            suggestion="Adoucissez les critères de filtrage"
                        ))
                    elif n_remaining < len(self.df) * 0.01:  # Moins de 1% restant
                        self.issues.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            code=f"FILTRE_TRES_RESTRICTIF_{i}",
                            message=f"Le filtre #{i+1} ne laisse que {n_remaining:,} lignes ({n_remaining/len(self.df):.1%})",
                            suggestion="Vérifiez que ce filtrage est intentionnel"
                        ))
                except Exception:
                    pass  # On ignore les erreurs de prévisualisation
    
    def _validate_coherence(self, plan: PlanAction):
        """Validation de cohérence globale du plan."""
        # Vérifier les conflits d'intention
        if plan.intention == IntentionType.VISUALISATION and plan.action == ActionType.CORRELATION:
            if len(plan.target_columns) > 5:
                self.issues.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    code="MATRICE_CORRELATION_TROP_GRANDE",
                    message=f"Matrice de corrélation avec {len(plan.target_columns)} colonnes",
                    suggestion="Envisagez de sélectionner moins de colonnes pour une meilleure lisibilité"
                ))
        
        # Vérifier la mémoire pour les grosses opérations
        estimated_memory = self._estimate_memory_usage(plan)
        if estimated_memory > 1e9:  # > 1GB
            self.issues.append(ValidationResult(
                level=ValidationLevel.WARNING,
                code="OPERATION_GOURMANDE_MEMOIRE",
                message=f"Opération potentiellement gourmande en mémoire (~{estimated_memory/1e9:.1f} GB estimés)",
                suggestion="Envisagez d'échantillonner les données ou d'optimiser l'opération"
            ))
        
        # Vérifier les actions redondantes
        if plan.action == ActionType.MOYENNE and len(plan.target_columns) == len(self.numeric_columns):
            self.issues.append(ValidationResult(
                level=ValidationLevel.INFO,
                code="ANALYSE_TOUTES_COLONNES",
                message="Analyse de toutes les colonnes numériques demandée",
                suggestion="Cela peut être intentionnel, mais vérifiez que vous n'avez pas besoin de filtrer"
            ))
    
    def _apply_filter_mask(self, col: str, op: str, val) -> pd.Series:
        """Applique un filtre pour vérification (sans modifier le dataframe)."""
        series = self.df[col]
        
        if op == '==':
            return series == val
        elif op == '!=':
            return series != val
        elif op == '>':
            return series > val
        elif op == '<':
            return series < val
        elif op == '>=':
            return series >= val
        elif op == '<=':
            return series <= val
        elif op == 'in':
            return series.isin(val if isinstance(val, list) else [val])
        elif op == 'not in':
            return ~series.isin(val if isinstance(val, list) else [val])
        elif op == 'contains':
            return series.astype(str).str.contains(str(val), na=False)
        elif op == 'startswith':
            return series.astype(str).str.startswith(str(val), na=False)
        elif op == 'endswith':
            return series.astype(str).str.endswith(str(val), na=False)
        else:
            return pd.Series([True] * len(self.df))
    
    def _estimate_memory_usage(self, plan: PlanAction) -> int:
        """Estime l'utilisation mémoire d'une opération en octets."""
        base_memory = self.df.memory_usage(deep=True).sum()
        
        multipliers = {
            ActionType.CORRELATION: 2.0,
            ActionType.REGRESSION: 3.0,
            ActionType.SCATTER_PLOT: 1.5,
            ActionType.DETECTER_OUTLIERS: 1.2,
            ActionType.GROUPBY_AGG: 2.5,
        }
        
        multiplier = multipliers.get(plan.action, 1.0)
        
        # Ajouter un facteur pour le groupby
        if plan.groupby_columns:
            multiplier *= 1.5
        
        return int(base_memory * multiplier)


# Fonction utilitaire pour validation rapide
def validate_plan(df: pd.DataFrame, plan: PlanAction) -> Tuple[bool, List[ValidationResult]]:
    """
    Valide un plan d'action de manière simple.
    
    Args:
        df: DataFrame à analyser
        plan: Plan d'action à valider
    
    Returns:
        Tuple (est_valide, liste_des_problèmes)
    """
    validator = PlanValidator(df)
    return validator.validate(plan)


def format_validation_results(results: List[ValidationResult]) -> str:
    """
    Formate les résultats de validation pour affichage.
    
    Args:
        results: Liste des résultats de validation
    
    Returns:
        Chaîne formatée
    """
    if not results:
        return "✅ Aucun problème détecté"
    
    lines = []
    for r in results:
        icon = "ℹ️" if r.level == ValidationLevel.INFO else "⚠️" if r.level == ValidationLevel.WARNING else "❌"
        lines.append(f"{icon} [{r.code}] {r.message}")
        if r.suggestion:
            lines.append(f"   💡 {r.suggestion}")
    
    return "\n".join(lines)
