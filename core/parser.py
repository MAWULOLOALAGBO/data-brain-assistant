"""
Module de parsing intelligent des requêtes utilisateur.
Transforme une question en langage naturel en plan d'action structuré.
"""

import re
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class IntentionType(Enum):
    STATISTIQUE = "statistique"
    VISUALISATION = "visualisation"
    NETTOYAGE = "nettoyage"
    EXPLORATION = "exploration"
    PREDICTION = "prediction"
    INCONNU = "inconnu"


class ActionType(Enum):
    # Statistiques
    MOYENNE = "moyenne"
    SOMME = "somme"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    MEDIANE = "mediane"
    ECART_TYPE = "ecart_type"
    VARIANCE = "variance"
    PERCENTILE = "percentile"
    COUNT = "count"
    
    # Visualisations
    HISTOGRAMME = "histogramme"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    BOX_PLOT = "box_plot"
    HEATMAP = "heatmap"
    
    # Nettoyage
    DETECTER_MANQUANTS = "detecter_manquants"
    DETECTER_DOUBLONS = "detecter_doublons"
    DETECTER_OUTLIERS = "detecter_outliers"
    REMPLACER_MANQUANTS = "remplacer_manquants"
    
    # Exploration
    DESCRIBE = "describe"
    DTYPES = "dtypes"
    CORRELATION = "correlation"
    HEAD = "head"
    TAIL = "tail"
    
    # Inconnu
    INCONNU = "inconnu"


@dataclass
class PlanAction:
    """Plan d'action structuré résultant du parsing."""
    intention: IntentionType
    action: ActionType
    target_columns: List[str]
    groupby_column: Optional[str] = None
    filter_conditions: List[Dict[str, Any]] = None
    parameters: Dict[str, Any] = None
    confidence: float = 0.0  # 0.0 à 1.0
    explanation: str = ""
    
    def __post_init__(self):
        if self.filter_conditions is None:
            self.filter_conditions = []
        if self.parameters is None:
            self.parameters = {}


# Dictionnaire de synonymes étendu
SYNONYMS = {
    # Statistiques
    'moyenne': ['mean', 'average', 'avg', 'moyen', 'moyenne'],
    'somme': ['sum', 'total', 'somme', 'addition'],
    'minimum': ['min', 'minimum', 'plus petit', 'plus bas', 'inf'],
    'maximum': ['max', 'maximum', 'plus grand', 'plus haut', 'sup'],
    'mediane': ['median', 'mediane', 'milieu'],
    'ecart_type': ['std', 'ecart-type', 'ecart type', 'standard', 'dispersion'],
    'variance': ['var', 'variance', 'variabilite'],
    'count': ['count', 'nombre', 'occurrences', 'frequence'],
    
    # Visualisations
    'histogramme': ['hist', 'histogramme', 'distribution', 'repartition', 'densite'],
    'bar_chart': ['bar', 'barplot', 'bar chart', 'diagramme barre', 'batons'],
    'scatter_plot': ['scatter', 'nuage', 'points', 'correlation', 'relation', 'xy'],
    'line_chart': ['line', 'ligne', 'courbe', 'evolution', 'temporel', 'serie'],
    'pie_chart': ['pie', 'camembert', 'secteur', 'circulaire', 'proportion'],
    'box_plot': ['box', 'boite', 'boxplot', 'moustache', 'quartiles'],
    'heatmap': ['heatmap', 'matrice', 'correlation matrix', 'chaleur'],
    
    # Nettoyage
    'detecter_manquants': ['manquant', 'missing', 'null', 'na', 'vide', 'absent'],
    'detecter_doublons': ['doublon', 'duplicate', 'dupli', 'identique', 'redondant'],
    'detecter_outliers': ['outlier', 'anomalie', 'aberrant', 'extreme', 'atypique'],
    
    # Exploration
    'describe': ['describe', 'resume', 'apercu', 'synthese', 'statistiques'],
    'dtypes': ['type', 'dtype', 'datatype', 'structure', 'format'],
    'correlation': ['correlation', 'correle', 'lie', 'dependance'],
    
    # Groupby
    'par': ['par', 'group', 'groupe', 'selon', 'pour chaque', 'chaque'],
    
    # Filtres
    'superieur': ['superieur', '>', 'plus grand que', 'au dessus', 'above'],
    'inferieur': ['inferieur', '<', 'plus petit que', 'au dessous', 'below'],
    'egal': ['egal', '=', 'vaut', 'est'],
}


def normalize_text(text: str) -> str:
    """Normalise le texte pour le parsing."""
    # Minuscules, suppression accents basique
    text = text.lower()
    # Remplacer caractères spéciaux par espaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Espaces multiples -> simple
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_column_candidates(query: str, columns: List[str]) -> List[Tuple[str, float]]:
    """
    Trouve toutes les colonnes candidates avec un score de confiance.
    Retourne liste triée par score décroissant.
    """
    query_norm = normalize_text(query)
    query_words = set(query_norm.split())
    candidates = []
    
    for col in columns:
        col_norm = normalize_text(col)
        col_words = set(col_norm.split())
        
        # Score 1 : correspondance exacte
        if col_norm == query_norm:
            candidates.append((col, 1.0))
            continue
        
        # Score 2 : colonne dans la requête
        if col_norm in query_norm:
            candidates.append((col, 0.9))
            continue
        
        # Score 3 : mots communs
        common_words = query_words & col_words
        if common_words:
            score = len(common_words) / max(len(query_words), len(col_words))
            candidates.append((col, score * 0.8))
            continue
        
        # Score 4 : synonymes
        for word in query_words:
            for key, synonyms in SYNONYMS.items():
                if word in synonyms:
                    # Vérifier si le synonyme correspond à la colonne
                    if any(syn in col_norm for syn in [key] + synonyms):
                        candidates.append((col, 0.6))
                        break
    
    # Dédoublonner et trier
    seen = set()
    unique_candidates = []
    for col, score in sorted(candidates, key=lambda x: x[1], reverse=True):
        if col not in seen:
            seen.add(col)
            unique_candidates.append((col, score))
    
    return unique_candidates


def detect_intention(query: str) -> Tuple[IntentionType, ActionType, float]:
    """
    Détecte l'intention et l'action à partir de la requête.
    Retourne (intention, action, confiance).
    """
    query_norm = normalize_text(query)
    
    # Mapping intention -> mots-clés
    intention_keywords = {
        IntentionType.STATISTIQUE: [
            'moyenne', 'mean', 'average', 'somme', 'sum', 'total', 'min', 'max',
            'mediane', 'median', 'ecart', 'std', 'variance', 'count', 'nombre'
        ],
        IntentionType.VISUALISATION: [
            'histogramme', 'hist', 'graphique', 'plot', 'chart', 'figure',
            'bar', 'scatter', 'nuage', 'ligne', 'line', 'camembert', 'pie',
            'box', 'boite', 'heatmap', 'matrice'
        ],
        IntentionType.NETTOYAGE: [
            'manquant', 'missing', 'null', 'doublon', 'duplicate', 'outlier',
            'anomalie', 'aberrant', 'nettoyer', 'clean'
        ],
        IntentionType.EXPLORATION: [
            'describe', 'resume', 'apercu', 'type', 'dtype', 'structure',
            'correlation', 'correle', 'explore', 'analyse'
        ],
    }
    
    # Compter les mots-clés par intention
    scores = {}
    for intention, keywords in intention_keywords.items():
        score = sum(1 for kw in keywords if kw in query_norm)
        scores[intention] = score
    
    # Intention gagnante
    best_intention = max(scores, key=scores.get)
    if scores[best_intention] == 0:
        best_intention = IntentionType.INCONNU
    
    # Détecter l'action spécifique
    action = detect_action(query_norm, best_intention)
    
    # Confiance basée sur le score relatif
    total_score = sum(scores.values())
    confidence = scores[best_intention] / total_score if total_score > 0 else 0.0
    
    return best_intention, action, confidence


def detect_action(query: str, intention: IntentionType) -> ActionType:
    """Détecte l'action spécifique selon l'intention."""
    
    if intention == IntentionType.STATISTIQUE:
        if any(w in query for w in SYNONYMS['moyenne']):
            return ActionType.MOYENNE
        elif any(w in query for w in SYNONYMS['somme']):
            return ActionType.SOMME
        elif any(w in query for w in SYNONYMS['minimum']):
            return ActionType.MINIMUM
        elif any(w in query for w in SYNONYMS['maximum']):
            return ActionType.MAXIMUM
        elif any(w in query for w in SYNONYMS['mediane']):
            return ActionType.MEDIANE
        elif any(w in query for w in SYNONYMS['ecart_type']):
            return ActionType.ECART_TYPE
        elif any(w in query for w in SYNONYMS['variance']):
            return ActionType.VARIANCE
        elif any(w in query for w in SYNONYMS['count']):
            return ActionType.COUNT
    
    elif intention == IntentionType.VISUALISATION:
        if any(w in query for w in SYNONYMS['histogramme']):
            return ActionType.HISTOGRAMME
        elif any(w in query for w in SYNONYMS['bar_chart']):
            return ActionType.BAR_CHART
        elif any(w in query for w in SYNONYMS['scatter_plot']):
            return ActionType.SCATTER_PLOT
        elif any(w in query for w in SYNONYMS['line_chart']):
            return ActionType.LINE_CHART
        elif any(w in query for w in SYNONYMS['pie_chart']):
            return ActionType.PIE_CHART
        elif any(w in query for w in SYNONYMS['box_plot']):
            return ActionType.BOX_PLOT
        elif any(w in query for w in SYNONYMS['heatmap']):
            return ActionType.HEATMAP
    
    elif intention == IntentionType.NETTOYAGE:
        if any(w in query for w in SYNONYMS['detecter_manquants']):
            return ActionType.DETECTER_MANQUANTS
        elif any(w in query for w in SYNONYMS['detecter_doublons']):
            return ActionType.DETECTER_DOUBLONS
        elif any(w in query for w in SYNONYMS['detecter_outliers']):
            return ActionType.DETECTER_OUTLIERS
    
    elif intention == IntentionType.EXPLORATION:
        if any(w in query for w in SYNONYMS['describe']):
            return ActionType.DESCRIBE
        elif any(w in query for w in SYNONYMS['dtypes']):
            return ActionType.DTYPES
        elif any(w in query for w in SYNONYMS['correlation']):
            return ActionType.CORRELATION
    
    return ActionType.INCONNU


def detect_groupby(query: str, columns: List[str]) -> Optional[str]:
    """
    Détecte si la requête contient un groupby.
    Cherche les patterns : "par X", "groupé par X", "pour chaque X"
    """
    patterns = [
        r'par\s+(\w+)',
        r'groupe\s+par\s+(\w+)',
        r'pour\s+chaque\s+(\w+)',
        r'selon\s+(\w+)',
        r'chaque\s+(\w+)',
        r'group\s+by\s+(\w+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            candidate = match.group(1).lower()
            # Vérifier si c'est une colonne valide
            for col in columns:
                if normalize_text(col) == candidate or candidate in normalize_text(col):
                    return col
    
    return None


def detect_filters(query: str, columns: List[str]) -> List[Dict[str, Any]]:
    """
    Détecte les conditions de filtrage dans la requête.
    Ex: "supérieur à 10", "égal à Paris", "entre 5 et 10"
    """
    filters = []
    
    # Pattern : supérieur / inférieur
    patterns_superieur = [
        (r'(superieur|>|plus\s+grand\s+que|au\s+dessus\s+de)\s+(\d+(?:\.\d+)?)', '>'),
        (r'(inferieur|<|plus\s+petit\s+que|au\s+dessous\s+de)\s+(\d+(?:\.\d+)?)', '<'),
        (r'(egal\s+a|vaut|est)\s+(\w+)', '=='),
        (r'entre\s+(\d+(?:\.\d+)?)\s+et\s+(\d+(?:\.\d+)?)', 'between'),
    ]
    
    for pattern, operator in patterns_superieur:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            if operator == 'between':
                filters.append({
                    'operator': 'between',
                    'value_min': float(match.group(1)),
                    'value_max': float(match.group(2))
                })
            else:
                filters.append({
                    'operator': operator,
                    'value': match.group(2)
                })
    
    return filters


def parse_query(query: str, df: pd.DataFrame) -> PlanAction:
    """
    Fonction principale : parse une requête et retourne un plan d'action.
    
    Args:
        query: Requête utilisateur en langage naturel
        df: DataFrame sur lequel opérer
    
    Returns:
        PlanAction structuré
    """
    columns = list(df.columns)
    query_norm = normalize_text(query)
    
    # 1. Détecter l'intention et l'action
    intention, action, conf_intention = detect_intention(query)
    
    # 2. Détecter les colonnes cibles
    column_candidates = find_column_candidates(query, columns)
    
    # Sélectionner les meilleures colonnes
    target_columns = []
    if column_candidates:
        # Prendre la meilleure correspondance
        best_score = column_candidates[0][1]
        target_columns = [col for col, score in column_candidates if score >= best_score * 0.7][:2]
    
    # Si aucune colonne détectée mais intention claire
    if not target_columns and intention != IntentionType.INCONNU:
        # Colonnes par défaut selon l'intention
        if intention in [IntentionType.STATISTIQUE, IntentionType.VISUALISATION]:
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                target_columns = [num_cols[0]]
    
    # 3. Détecter le groupby
    groupby_col = detect_groupby(query, columns)
    
    # 4. Détecter les filtres
    filters = detect_filters(query, columns)
    
    # 5. Calculer la confiance globale
    conf_columns = column_candidates[0][1] if column_candidates else 0.0
    confidence = (conf_intention * 0.5 + conf_columns * 0.5)
    
    # 6. Générer l'explication
    explanation = generate_explanation(intention, action, target_columns, groupby_col)
    
    return PlanAction(
        intention=intention,
        action=action,
        target_columns=target_columns,
        groupby_column=groupby_col,
        filter_conditions=filters,
        confidence=confidence,
        explanation=explanation
    )


def generate_explanation(
    intention: IntentionType,
    action: ActionType,
    target_columns: List[str],
    groupby_col: Optional[str]
) -> str:
    """Génère une explication lisible du plan d'action."""
    
    parts = []
    
    # Intention
    intention_str = {
        IntentionType.STATISTIQUE: "calcul statistique",
        IntentionType.VISUALISATION: "visualisation",
        IntentionType.NETTOYAGE: "analyse de qualité des données",
        IntentionType.EXPLORATION: "exploration des données",
        IntentionType.INCONNU: "analyse"
    }.get(intention, "analyse")
    
    parts.append(f"{intention_str}")
    
    # Action
    if action != ActionType.INCONNU:
        parts.append(f"({action.value})")
    
    # Colonnes
    if target_columns:
        cols_str = ", ".join(target_columns)
        parts.append(f"sur {cols_str}")
    
    # Groupby
    if groupby_col:
        parts.append(f"groupé par '{groupby_col}'")
    
    return " ".join(parts)


# Fonction utilitaire pour debugging
def debug_parse(query: str, df: pd.DataFrame) -> Dict:
    """Affiche les étapes intermédiaires du parsing."""
    columns = list(df.columns)
    
    return {
        'query': query,
        'normalized': normalize_text(query),
        'intention': detect_intention(query),
        'columns_candidates': find_column_candidates(query, columns),
        'groupby': detect_groupby(query, columns),
        'filters': detect_filters(query, columns),
        'plan': parse_query(query, df)
    }
