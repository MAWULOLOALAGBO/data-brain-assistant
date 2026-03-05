"""
Data Brain Assistant - Interface Streamlit finale
Architecture modulaire, sans exec(), 100% sécurisée
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Ajout du core au path
sys.path.insert(0, str(Path(__file__).parent))

from core.loader import load_file, infer_types, get_column_info
from core.parser import parse_query, PlanAction, IntentionType, ActionType
from core.validator import validate_plan, ValidationLevel
from core.executor import execute_action, ExecutionError
from core.actions import (
    STATS_AVAILABLE, VIZ_AVAILABLE, 
    CLEANING_AVAILABLE, EXPLORATION_AVAILABLE
)

# Configuration de la page
st.set_page_config(
    page_title="🧠 Data Brain Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .stPlotlyChart {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== INITIALISATION SESSION ====================

def init_session():
    """Initialise les variables de session."""
    defaults = {
        'data': None,
        'metadata': None,
        'history': [],
        'last_plan': None,
        'last_result': None,
        'show_advanced': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ==================== SIDEBAR ====================

def render_sidebar():
    """Rend la barre latérale."""
    with st.sidebar:
        st.title("🧠 Data Brain")
        st.markdown("*Assistant d'analyse universel*")
        st.divider()
        
        # Historique
        if st.session_state.history:
            st.header("📜 Historique")
            for i, item in enumerate(reversed(st.session_state.history[-8:])):
                with st.expander(f"{item['time']} - {item['query'][:30]}...", expanded=False):
                    st.caption(f"Intention: {item['intention']}")
                    st.caption(f"Action: {item['action']}")
                    if item.get('success'):
                        st.success("✅ Succès")
                    else:
                        st.error("❌ Échec")
            
            if st.button("🗑️ Effacer l'historique", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        
        st.divider()
        
        # Aide contextuelle
        st.header("❓ Aide")
        
        with st.expander("Types de requêtes"):
            st.markdown("""
            **Statistiques:** moyenne, somme, minimum, maximum, médiane, écart-type
            
            **Visualisations:** histogramme, bar, scatter, line, pie, box, heatmap
            
            **Nettoyage:** valeurs manquantes, doublons, outliers
            
            **Exploration:** describe, types, corrélation, profil
            """)
        
        with st.expander("Syntaxe"):
            st.markdown("""
            - *"moyenne de Age par Gender"*
            - *"histogramme des prix"*
            - *"corrélations entre variables numériques"*
            - *"détecter les valeurs manquantes"*
            """)


# ==================== CHARGEMENT DE FICHIER ====================

def render_upload_section():
    """Section de chargement de fichier."""
    st.header("📁 1. Chargez vos données")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Déposez un fichier (CSV, Excel, JSON, Parquet)",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'txt'],
            help="Formats supportés: CSV, Excel (.xlsx, .xls), JSON, Parquet, TXT"
        )
    
    with col2:
        st.info("""
        **Formats auto-détectés:**
        - Encodage (UTF-8, Latin-1, etc.)
        - Séparateurs (virgule, point-virgule, tab)
        - Types de données
        """)
    
    if uploaded_file is None:
        # Fichier exemple pour démo
        st.info("👆 Chargez un fichier ou utilisez l'exemple ci-dessous")
        
        if st.button("📊 Charger l'exemple (Données de vente)"):
            # Créer un DataFrame exemple
            import numpy as np
            np.random.seed(42)
            n = 1000
            
            example_df = pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=n, freq='D'),
                'Produit': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
                'Region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], n),
                'Quantite': np.random.randint(1, 100, n),
                'Prix_Unitaire': np.random.uniform(10, 500, n).round(2),
                'Client_Satisfait': np.random.choice([True, False], n, p=[0.8, 0.2])
            })
            example_df['Revenu'] = (example_df['Quantite'] * example_df['Prix_Unitaire']).round(2)
            
            st.session_state.data = example_df
            st.session_state.metadata = {
                'filename': 'exemple_ventes.csv',
                'detected_format': 'CSV (généré)',
                'rows': len(example_df),
                'columns': len(example_df.columns)
            }
            st.rerun()
        
        return False
    
    # Traitement du fichier uploadé
    try:
        with st.spinner("🔍 Analyse du fichier..."):
            df, metadata = load_file(uploaded_file, uploaded_file.name)
            
            # Inférence de types (non-aggressive par défaut)
            df = infer_types(df, aggressive=False)
            
            st.session_state.data = df
            st.session_state.metadata = metadata
            
        # Affichage du succès
        st.success(f"""
        ✅ **{metadata['filename']}** chargé  
        **{metadata['rows']:,}** lignes × **{metadata['columns']}** colonnes  
        Format: {metadata['detected_format']} | 
        Mémoire: {metadata.get('size_bytes', 0)/1024/1024:.2f} MB
        """)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {str(e)}")
        st.info("💡 Essayez de vérifier le format ou l'encodage du fichier")
        return False


# ==================== APERÇU DES DONNÉES ====================

def render_data_preview():
    """Aperçu et métadonnées du dataset."""
    df = st.session_state.data
    if df is None:
        return
    
    st.header("🔍 2. Aperçu des données")
    
    tabs = st.tabs(["📋 Données", "📊 Structure", "📈 Statistiques rapides"])
    
    with tabs[0]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df.head(100), use_container_width=True, height=400)
        with col2:
            st.metric("Total lignes", len(df))
            st.metric("Total colonnes", len(df.columns))
            st.metric("Mémoire (MB)", round(df.memory_usage(deep=True).sum() / 1024**2, 2))
            
            # Bouton téléchargement
            csv = df.head(1000).to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Télécharger (CSV)",
                csv,
                "data_preview.csv",
                "text/csv"
            )
    
    with tabs[1]:
        col_info = get_column_info(df)
        st.dataframe(col_info, use_container_width=True)
        
        # Types détectés
        type_counts = df.dtypes.value_counts()
        st.caption("Répartition des types pandas:")
        st.write(dict(type_counts))
    
    with tabs[2]:
        # Stats rapides sur colonnes numériques
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            st.dataframe(df[num_cols].describe().transpose(), use_container_width=True)
        else:
            st.info("Pas de colonnes numériques pour les statistiques")


# ==================== SECTION REQUÊTE ====================

def render_query_section():
    """Section de requête utilisateur."""
    df = st.session_state.data
    if df is None:
        return
    
    st.header("💬 3. Posez votre question")
    
    # Suggestions contextuelles
    st.caption("Suggestions basées sur vos données:")
    
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    suggestions = []
    if num_cols:
        suggestions.append(f"moyenne de {num_cols[0]}")
        suggestions.append(f"histogramme de {num_cols[0]}")
    if num_cols and cat_cols:
        suggestions.append(f"moyenne de {num_cols[0]} par {cat_cols[0]}")
    if len(num_cols) >= 2:
        suggestions.append(f"scatter de {num_cols[0]} et {num_cols[1]}")
    suggestions.append("profil complet des données")
    
    cols = st.columns(min(len(suggestions), 4))
    for i, sugg in enumerate(suggestions[:4]):
        with cols[i]:
            if st.button(f"💡 {sugg[:25]}...", key=f"sugg_{i}", use_container_width=True):
                st.session_state.suggestion = sugg
                st.rerun()
    
    # Input utilisateur
    default_query = st.session_state.get('suggestion', '')
    query = st.text_input(
        "Votre question en langage naturel:",
        value=default_query,
        placeholder="Ex: 'moyenne des revenus par région', 'corrélations', 'valeurs manquantes'...",
        key="query_input"
    )
    
    # Options avancées
    with st.expander("⚙️ Options avancées"):
        st.session_state.show_advanced = True
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Mode strict (refuse les ambiguïtés)", value=False, key="strict_mode")
        with col2:
            st.checkbox("Afficher le plan détaillé", value=True, key="show_plan")
    
    if not query:
        return None
    
    return query


# ==================== EXÉCUTION ET RÉSULTATS ====================

def render_execution(query: str):
    """Exécute la requête et affiche les résultats."""
    df = st.session_state.data
    
    st.divider()
    st.header("🚀 4. Résultat de l'analyse")
    
    # ÉTAPE 1: Parsing
    with st.spinner("🧠 Compréhension de la requête..."):
        try:
            plan = parse_query(query, df)
            st.session_state.last_plan = plan
        except Exception as e:
            st.error(f"❌ Erreur de compréhension: {str(e)}")
            return
    
    # Affichage du plan (si option activée)
    if st.session_state.get('show_plan', True):
        with st.expander("📋 Plan d'action détecté", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**Intention:** `{plan.intention.value}`")
            with col2:
                st.markdown(f"**Action:** `{plan.action.value}`")
            with col3:
                cols_str = ", ".join(plan.target_columns[:3])
                if len(plan.target_columns) > 3:
                    cols_str += "..."
                st.markdown(f"**Colonnes:** `{cols_str}`")
            with col4:
                st.markdown(f"**Groupby:** `{plan.groupby_column or 'Aucun'}`")
            
            st.progress(min(plan.confidence, 1.0), text=f"Confiance: {plan.confidence:.0%}")
            st.info(f"💡 {plan.explanation}")
    
    # ÉTAPE 2: Validation
    with st.spinner("✅ Validation du plan..."):
        is_valid, issues = validate_plan(plan, df)
        
        # Affichage des problèmes
        errors = [i for i in issues if i.level == ValidationLevel.ERROR]
        warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
        infos = [i for i in issues if i.level == ValidationLevel.INFO]
        
        if errors:
            st.error("### ❌ Problèmes bloquants")
            for issue in errors:
                with st.container():
                    st.markdown(f"**{issue.code}**: {issue.message}")
                    if issue.suggestion:
                        st.caption(f"💡 Suggestion: {issue.suggestion}")
            
            # Mode strict = blocage total
            if st.session_state.get('strict_mode', False):
                st.error("🚫 Exécution bloquée (mode strict)")
                _log_history(query, plan, False, "Validation échouée")
                return
        
        if warnings:
            with st.expander(f"⚠️ {len(warnings)} avertissement(s)"):
                for issue in warnings:
                    st.markdown(f"**{issue.code}**: {issue.message}")
                    if issue.suggestion:
                        st.caption(f"💡 {issue.suggestion}")
        
        if infos and st.session_state.get('show_advanced', False):
            with st.expander(f"ℹ️ {len(infos)} information(s)"):
                for issue in infos:
                    st.markdown(f"**{issue.code}**: {issue.message}")
    
    # ÉTAPE 3: Exécution
    with st.spinner("⚡ Exécution..."):
        try:
            result, fig, metadata = execute_action(plan, df)
            st.session_state.last_result = result
            
            # Affichage du résultat
            _render_result(result, fig, plan)
            
            _log_history(query, plan, True, "Succès")
            
        except ExecutionError as e:
            st.error(f"❌ Erreur d'exécution: {str(e)}")
            _log_history(query, plan, False, str(e))
            
        except Exception as e:
            st.error(f"💥 Erreur inattendue: {str(e)}")
            st.exception(e)
            _log_history(query, plan, False, f"Exception: {str(e)}")


def _render_result(result, fig, plan: PlanAction):
    """Affiche le résultat selon son type."""
    
    # Visualisation = graphique prioritaire
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, use_container_height=False)
        
        # Téléchargement du graphique
        img_bytes = fig.to_image(format="png", scale=2)
        st.download_button(
            "📷 Télécharger le graphique (PNG)",
            img_bytes,
            f"graph_{plan.action.value}.png",
            "image/png"
        )
    
    # Résultat texte/dataframe
    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True)
        
        if len(result) > 10:
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Télécharger les résultats (CSV)",
                csv,
                f"result_{plan.action.value}.csv",
                "text/csv"
            )
    
    elif isinstance(result, pd.Series):
        if len(result) <= 20:
            # Série courte = bar chart
            st.bar_chart(result)
        else:
            st.dataframe(result)
    
    elif isinstance(result, dict):
        st.json(result)
    
    elif isinstance(result, str):
        if result.startswith("✅"):
            st.success(result)
        elif result.startswith("⚠️"):
            st.warning(result)
        else:
            st.info(result)
    
    else:
        st.write(result)


def _log_history(query: str, plan: PlanAction, success: bool, detail: str):
    """Log dans l'historique."""
    from datetime import datetime
    st.session_state.history.append({
        'time': datetime.now().strftime("%H:%M"),
        'query': query,
        'intention': plan.intention.value,
        'action': plan.action.value,
        'success': success,
        'detail': detail
    })


# ==================== MAIN ====================

def main():
    """Fonction principale."""
    render_sidebar()
    
    # Titre
    st.markdown('<p class="main-header">🧠 Data Brain Assistant</p>', unsafe_allow_html=True)
    st.caption("Analysez n'importe quel fichier avec une simple phrase en langage naturel")
    st.divider()
    
    # Étape 1: Chargement
    file_loaded = render_upload_section()
    
    if not file_loaded or st.session_state.data is None:
        # Page d'accueil / instructions
        st.info("""
        ### 👋 Bienvenue !
        
        **Data Brain Assistant** est un outil d'analyse de données universel :
        
        1. **Chargez** n'importe quel fichier (CSV, Excel, JSON...)
        2. **Posez une question** en français ou anglais
        3. **Obtenez** instantanément des statistiques, graphiques, analyses
        
        **Exemples de requêtes:**
        - *"moyenne des salaires par département"*
        - *"histogramme des âges"*
        - *"corrélations entre toutes les variables numériques"*
        - *"détecter les valeurs manquantes"*
        """)
        return
    
    # Étape 2: Aperçu
    render_data_preview()
    
    # Étape 3: Requête
    query = render_query_section()
    
    # Étape 4: Exécution
    if query:
        render_execution(query)


if __name__ == "__main__":
    main()
