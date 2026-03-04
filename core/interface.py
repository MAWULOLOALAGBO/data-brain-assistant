"""
Module d'interface utilisateur.
Point d'entrée unique pour interagir avec le système d'analyse de données.
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

from .parser import parse_query, PlanAction, IntentionType, ActionType
from .validator import PlanValidator, ValidationResult, ValidationLevel, validate_plan, format_validation_results
from .executor import PlanExecutor, ExecutionResult, ExecutionStatus, execute_plan


class ProcessingMode(Enum):
    """Modes de traitement disponibles."""
    STRICT = "strict"       # Stoppe si validation échoue
    PERMISSIVE = "permissive"  # Exécute malgré les warnings
    VALIDATE_ONLY = "validate_only"  # Uniquement valider, pas exécuter


@dataclass
class SystemResponse:
    """Réponse standardisée du système."""
    success: bool
    query: str
    intention: str
    action: str
    validation_passed: bool
    validation_issues: List[Dict]
    execution_status: str
    result: Optional[Dict] = None
    message: str = ""
    execution_time_ms: float = 0.0
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour JSON."""
        return {
            'success': self.success,
            'query': self.query,
            'intention': self.intention,
            'action': self.action,
            'validation_passed': self.validation_passed,
            'validation_issues': self.validation_issues,
            'execution_status': self.execution_status,
            'result': self._serialize_result(),
            'message': self.message,
            'execution_time_ms': self.execution_time_ms,
            'suggestions': self.suggestions
        }
    
    def _serialize_result(self):
        """Sérialise le résultat pour JSON."""
        if self.result is None:
            return None
        
        result_data = self.result.get('data') if isinstance(self.result, dict) else self.result
        
        if isinstance(result_data, pd.DataFrame):
            return {
                'type': 'dataframe',
                'data': result_data.head(20).to_dict(orient='records'),
                'shape': list(result_data.shape),
                'columns': list(result_data.columns)
            }
        elif isinstance(result_data, pd.Series):
            return {
                'type': 'series',
                'data': result_data.head(20).to_dict(),
                'length': len(result_data)
            }
        elif isinstance(result_data, dict) and 'image_base64' in result_data:
            return {
                'type': 'visualization',
                'image_base64': result_data['image_base64'],
                'format': 'png'
            }
        else:
            return {
                'type': 'value',
                'data': result_data
            }
    
    def to_json(self) -> str:
        """Exporte en JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class DataAnalysisSystem:
    """
    Système complet d'analyse de données par langage naturel.
    Point d'entrée principal pour toutes les opérations.
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[Dict] = None):
        """
        Initialise le système avec un dataset.
        
        Args:
            df: DataFrame pandas à analyser
            config: Configuration optionnelle
        """
        self.original_df = df.copy()
        self.working_df = df.copy()
        self.config = config or {}
        self.history = []
        self.validator = PlanValidator(self.working_df)
        self.executor = PlanExecutor(self.working_df, config)
        
        # Statistiques du dataset
        self._compute_dataset_stats()
    
    def _compute_dataset_stats(self):
        """Calcule les statistiques initiales du dataset."""
        self.stats = {
            'total_rows': len(self.working_df),
            'total_columns': len(self.working_df.columns),
            'numeric_columns': self.working_df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.working_df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': self.working_df.select_dtypes(include=['datetime64']).columns.tolist(),
            'memory_usage_mb': self.working_df.memory_usage(deep=True).sum() / 1024**2
        }
    
    def ask(self, query: str, mode: ProcessingMode = ProcessingMode.STRICT) -> SystemResponse:
        """
        Point d'entrée principal : pose une question en langage naturel.
        
        Args:
            query: Question/requête en français
            mode: Mode de traitement (strict, permissif, validation seule)
        
        Returns:
            SystemResponse structurée
        """
        import time
        total_start = time.time()
        
        # Étape 1: Parsing
        try:
            plan = parse_query(query, self.working_df.columns.tolist())
        except Exception as e:
            return SystemResponse(
                success=False,
                query=query,
                intention="unknown",
                action="unknown",
                validation_passed=False,
                validation_issues=[{'level': 'error', 'message': f'Erreur parsing: {str(e)}'}],
                execution_status="failed",
                message=f"Impossible d'interpréter la requête: {str(e)}"
            )
        
        # Étape 2: Validation
        is_valid, issues = self.validator.validate(plan)
        validation_issues = [
            {
                'level': issue.level.value,
                'code': issue.code,
                'message': issue.message,
                'suggestion': issue.suggestion
            }
            for issue in issues
        ]
        
        # Vérifier s'il y a des erreurs bloquantes
        has_errors = any(i['level'] == 'error' for i in validation_issues)
        
        if mode == ProcessingMode.VALIDATE_ONLY:
            return SystemResponse(
                success=True,
                query=query,
                intention=plan.intention.value,
                action=plan.action.value,
                validation_passed=not has_errors,
                validation_issues=validation_issues,
                execution_status="validated_only",
                message="Validation terminée (exécution non demandée)"
            )
        
        if has_errors and mode == ProcessingMode.STRICT:
            return SystemResponse(
                success=False,
                query=query,
                intention=plan.intention.value,
                action=plan.action.value,
                validation_passed=False,
                validation_issues=validation_issues,
                execution_status="blocked",
                message="Exécution bloquée par des erreurs de validation",
                suggestions=[i['suggestion'] for i in validation_issues if i['suggestion']]
            )
        
        # Étape 3: Exécution
        try:
            exec_result = self.executor.execute(plan)
            
            # Mettre à jour le working_df si c'était une opération de nettoyage
            if plan.intention == IntentionType.NETTOYAGE and exec_result.status == ExecutionStatus.SUCCESS:
                self.working_df = self.executor.df
                self.validator = PlanValidator(self.working_df)  # Recréer le validateur
            
            # Construire la réponse
            total_time = (time.time() - total_start) * 1000
            
            response = SystemResponse(
                success=exec_result.status in [ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL],
                query=query,
                intention=plan.intention.value,
                action=plan.action.value,
                validation_passed=not has_errors,
                validation_issues=validation_issues,
                execution_status=exec_result.status.value,
                result={
                    'data': exec_result.data,
                    'metadata': exec_result.metadata,
                    'message': exec_result.message
                },
                message=exec_result.message,
                execution_time_ms=total_time + exec_result.execution_time_ms
            )
            
            # Ajouter des suggestions contextuelles
            response.suggestions = self._generate_suggestions(plan, exec_result, issues)
            
            # Historiser
            self.history.append({
                'query': query,
                'plan': plan,
                'response': response,
                'timestamp': pd.Timestamp.now()
            })
            
            return response
            
        except Exception as e:
            return SystemResponse(
                success=False,
                query=query,
                intention=plan.intention.value,
                action=plan.action.value,
                validation_passed=not has_errors,
                validation_issues=validation_issues,
                execution_status="failed",
                message=f"Erreur d'exécution: {str(e)}"
            )
    
    def _generate_suggestions(self, plan: PlanAction, result: ExecutionResult, 
                             issues: List[ValidationResult]) -> List[str]:
        """Génère des suggestions contextuelles basées sur le résultat."""
        suggestions = []
        
        # Suggestions basées sur l'intention actuelle
        if plan.intention == IntentionType.STATISTIQUE:
            if plan.action == ActionType.MOYENNE:
                suggestions.append("Essayez 'écart-type de [colonne]' pour voir la dispersion")
                suggestions.append("Essayez 'corrélation entre [col1] et [col2]' pour les relations")
        
        elif plan.intention == IntentionType.VISUALISATION:
            if plan.action == ActionType.HISTOGRAMME:
                suggestions.append("Essayez 'boxplot de [colonne]' pour voir les outliers")
            elif plan.action == ActionType.SCATTER_PLOT:
                suggestions.append("Essayez 'régression de [y] par [x]' pour modéliser la relation")
        
        # Suggestions si valeurs manquantes détectées
        missing_cols = self.working_df.columns[self.working_df.isnull().any()].tolist()
        if missing_cols and plan.intention != IntentionType.NETTOYAGE:
            suggestions.append(f"Note: {len(missing_cols)} colonne(s) ont des valeurs manquantes. Demandez 'nettoyer données' pour les traiter")
        
        # Suggestions de colonnes similaires si erreur de nom
        for issue in issues:
            if issue.code == "COLONNE_INEXISTANTE" and issue.suggestion:
                suggestions.append(f"Did you mean? {issue.suggestion}")
        
        return suggestions[:3]  # Max 3 suggestions
    
    def get_dataset_info(self) -> Dict:
        """Retourne les informations sur le dataset actuel."""
        return {
            'shape': self.working_df.shape,
            'columns': [
                {
                    'name': col,
                    'type': str(self.working_df[col].dtype),
                    'null_count': int(self.working_df[col].isnull().sum()),
                    'null_pct': float(self.working_df[col].isnull().mean() * 100),
                    'unique_count': int(self.working_df[col].nunique())
                }
                for col in self.working_df.columns
            ],
            'memory_usage_mb': round(self.working_df.memory_usage(deep=True).sum() / 1024**2, 2),
            'sample': self.working_df.head(3).to_dict(orient='records')
        }
    
    def reset_data(self):
        """Réinitialise le dataset à son état original."""
        self.working_df = self.original_df.copy()
        self.validator = PlanValidator(self.working_df)
        self.executor = PlanExecutor(self.working_df, self.config)
        self.history.clear()
        return "Dataset réinitialisé"
    
    def get_history(self) -> List[Dict]:
        """Retourne l'historique des requêtes."""
        return [
            {
                'query': h['query'],
                'intention': h['plan'].intention.value,
                'success': h['response'].success,
                'timestamp': h['timestamp'].isoformat()
            }
            for h in self.history
        ]
    
    def export_working_data(self, format: str = 'csv') -> Union[str, bytes]:
        """Exporte le dataset actuel."""
        if format == 'csv':
            return self.working_df.to_csv(index=False)
        elif format == 'json':
            return self.working_df.to_json(orient='records', indent=2)
        elif format == 'excel':
            buffer = io.BytesIO()
            self.working_df.to_excel(buffer, index=False)
            return buffer.getvalue()
        else:
            raise ValueError(f"Format {format} non supporté")


# Fonctions utilitaires de haut niveau

def analyze_dataframe(df: pd.DataFrame, query: str, verbose: bool = False) -> Dict:
    """
    Fonction simple pour analyser un DataFrame avec une requête.
    
    Args:
        df: DataFrame à analyser
        query: Requête en langage naturel
        verbose: Afficher les détails de validation
    
    Returns:
        Dictionnaire avec le résultat
    """
    system = DataAnalysisSystem(df)
    
    if verbose:
        print(f"🔍 Requête: {query}")
        print(f"📊 Dataset: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    response = system.ask(query)
    
    if verbose:
        print(f"\n{'✅' if response.success else '❌'} {response.message}")
        if response.validation_issues:
            print(f"\n⚠️  Problèmes détectés ({len(response.validation_issues)}):")
            for issue in response.validation_issues:
                icon = "❌" if issue['level'] == 'error' else "⚠️" if issue['level'] == 'warning' else "ℹ️"
                print(f"   {icon} {issue['message']}")
    
    return response.to_dict()


def interactive_session(df: pd.DataFrame):
    """
    Lance une session interactive dans le terminal.
    
    Args:
        df: DataFrame à explorer
    """
    system = DataAnalysisSystem(df)
    
    print("=" * 60)
    print("🤖 Assistant d'Analyse de Données")
    print("=" * 60)
    print(f"Dataset chargé: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print("Colonnes:", ", ".join(df.columns[:5]), "..." if len(df.columns) > 5 else "")
    print("\nCommandes spéciales:")
    print("  'info' - Info sur le dataset")
    print("  'reset' - Réinitialiser les données")
    print("  'history' - Voir l'historique")
    print("  'quit' - Quitter")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n📝 Votre question > ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Au revoir !")
                break
            
            if query.lower() == 'info':
                info = system.get_dataset_info()
                print(f"\n📊 Info Dataset:")
                print(f"   Dimensions: {info['shape']}")
                print(f"   Mémoire: {info['memory_usage_mb']} MB")
                print(f"\n   Colonnes:")
                for col in info['columns'][:5]:
                    null_info = f"{col['null_count']} nulls ({col['null_pct']:.1f}%)" if col['null_count'] > 0 else "complet"
                    print(f"   • {col['name']} ({col['type']}) - {null_info}")
                continue
            
            if query.lower() == 'reset':
                print(system.reset_data())
                continue
            
            if query.lower() == 'history':
                hist = system.get_history()
                print(f"\n📜 Historique ({len(hist)} requêtes):")
                for h in hist[-5:]:
                    status = "✅" if h['success'] else "❌"
                    print(f"   {status} [{h['intention']}] {h['query'][:50]}...")
                continue
            
            if not query:
                continue
            
            # Traiter la requête
            response = system.ask(query)
            
            # Afficher le résultat
            print(f"\n{'✅' if response.success else '❌'} {response.message}")
            
            if response.result:
                result_data = response.result.get('data')
                
                if isinstance(result_data, pd.DataFrame):
                    print(f"\n{result_data.to_string()}")
                elif isinstance(result_data, dict) and 'image_base64' in result_data:
                    print("📈 [Visualisation générée - voir dans interface graphique]")
                else:
                    print(f"Résultat: {result_data}")
            
            if response.suggestions:
                print(f"\n💡 Suggestions:")
                for sug in response.suggestions:
                    print(f"   • {sug}")
                    
        except KeyboardInterrupt:
            print("\n\nInterrompu. Tapez 'quit' pour sortir.")
        except Exception as e:
            print(f"\n💥 Erreur: {str(e)}")


# Export pour utilisation externe
__all__ = [
    'DataAnalysisSystem',
    'SystemResponse',
    'ProcessingMode',
    'analyze_dataframe',
    'interactive_session'
]
