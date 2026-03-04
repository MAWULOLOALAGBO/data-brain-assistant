import streamlit as st
import pandas as pd
import json
import re
import numpy as np
from io import StringIO

# ==================== CONFIGURATION ====================

st.set_page_config(page_title="Data Brain Assistant", layout="wide", page_icon="🧠")

# Initialisation session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'plan' not in st.session_state:
    st.session_state.plan = None

# ==================== FONCTIONS UTILITAIRES ====================

def infer_types(df):
    """Inférence intelligente des types de données"""
    df = df.copy()
    for col in df.columns:
        # Détection datetime
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                continue
            except:
                pass
            
            # Détection numérique
            try:
                df[col] = pd.to_numeric(df[col].str.replace(',', '.').str.replace(' ', ''))
                continue
            except:
                pass
                
            # Détection catégorielle
            if df[col].nunique() / len(df) < 0.05:
                df[col] = df[col].astype('category')
    
    return df

def generate_suggestions(df):
    """Génère des suggestions contextuelles basées sur les données"""
    suggestions = []
    
    # Suggestion 1 : Statistiques générales
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        suggestions.append(f"moyenne de {num_cols[0]}")
    
    # Suggestion 2 : Visualisation
    if len(num_cols) > 0:
        suggestions.append(f"histogramme de {num_cols[0]}")
    
    # Suggestion 3 : Groupby si colonnes catégorielles
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols and num_cols:
        suggestions.append(f"moyenne de {num_cols[0]} par {cat_cols[0]}")
    
    # Compléter si besoin
    while len(suggestions) < 3:
        suggestions.append("describe")
    
    return suggestions[:3]

def find_column(query, columns):
    """Trouve la meilleure correspondance de colonne"""
    query_lower = query.lower()
    
    # Correspondance exacte
    for col in columns:
        if col.lower() == query_lower:
            return col
    
    # Correspondance partielle
    for col in columns:
        if query_lower in col.lower() or col.lower() in query_lower:
            return col
    
    # Synonymes communs
    synonyms = {
        'genre': ['gender', 'sex', 'sexe'],
        'prix': ['price', 'prix', 'cost', 'coût', 'avg_price', 'revenue', 'revenu'],
        'revenu': ['revenue', 'income', 'salaire', 'salary', 'revenu'],
        'quantite': ['quantity', 'quantité', 'qty', 'units', 'units_sold', 'volume'],
        'age': ['age', 'âge', 'annee', 'year'],
        'region': ['region', 'région', 'area', 'zone', 'country', 'pays', 'ville', 'city'],
        'date': ['date', 'time', 'timestamp', 'datetime', 'mois', 'month', 'jour', 'day'],
        'nom': ['name', 'nom', 'id', 'identifier', 'user_id', 'client'],
        'categorie': ['category', 'categorie', 'catégorie', 'type', 'segment', 'model', 'modele']
    }
    
    for key, values in synonyms.items():
        if query_lower in key or key in query_lower:
            for col in columns:
                for syn in values:
                    if syn in col.lower():
                        return col
    
    return None

def parse_query_v2(query, df):
    """Parseur intelligent v2 avec groupby et synonymes"""
    query_lower = query.lower()
    columns = list(df.columns)
    
    # Détection groupby
    groupby_col = None
    groupby_match = re.search(r'par\s+(\w+)', query_lower)
    if groupby_match:
        groupby_candidate = groupby_match.group(1)
        groupby_col = find_column(groupby_candidate, columns)
    
    # Détection des colonnes cibles
    target_cols = []
    words = re.findall(r'\b\w+\b', query_lower)
    
    for word in words:
        if len(word) > 2:  # Ignorer les mots courts
            col = find_column(word, columns)
            if col and col not in target_cols:
                target_cols.append(col)
    
    # Détection intention
    intention = "exploration"
    action = "analyse générale"
    method = "describe()"
    
    # Statistiques
    if any(w in query_lower for w in ["moyenne", "mean", "moyen", "average"]):
        intention = "statistique"
        action = "calcul de la moyenne"
        method = "mean()"
    elif any(w in query_lower for w in ["somme", "total", "sum"]):
        intention = "statistique"
        action = "calcul de la somme"
        method = "sum()"
    elif any(w in query_lower for w in ["minimum", "min", "plus petit"]):
        intention = "statistique"
        action = "valeur minimale"
        method = "min()"
    elif any(w in query_lower for w in ["maximum", "max", "plus grand"]):
        intention = "statistique"
        action = "valeur maximale"
        method = "max()"
    elif any(w in query_lower for w in ["mediane", "median", "médiane"]):
        intention = "statistique"
        action = "calcul de la médiane"
        method = "median()"
    elif any(w in query_lower for w in ["ecart-type", "std", "standard", "variance", "dispersion"]):
        intention = "statistique"
        action = "calcul de l'écart-type"
        method = "std()"
    
    # Visualisation
    elif any(w in query_lower for w in ["histogramme", "hist", "distribution"]):
        intention = "visualisation"
        action = "histogramme de distribution"
        method = "px.histogram()"
    elif any(w in query_lower for w in ["barplot", "bar", "barres", "diagramme barre"]):
        intention = "visualisation"
        action = "diagramme en barres"
        method = "px.bar()"
    elif any(w in query_lower for w in ["camembert", "pie", "secteur", "cercle"]):
        intention = "visualisation"
        action = "diagramme circulaire"
        method = "px.pie()"
    elif any(w in query_lower for w in ["scatter", "nuage", "point", "correlation", "corrélation", "relation"]):
        intention = "visualisation"
        action = "nuage de points"
        method = "px.scatter()"
    elif any(w in query_lower for w in ["ligne", "line", "evolution", "évolution", "temporel"]):
        intention = "visualisation"
        action = "graphique linéaire"
        method = "px.line()"
    
    # Nettoyage
    elif any(w in query_lower for w in ["manquant", "missing", "null", "na", "vide", "absent"]):
        intention = "nettoyage"
        action = "détection des valeurs manquantes"
        method = "isnull().sum()"
    elif any(w in query_lower for w in ["doublon", "duplicate", "dupli", "identique"]):
        intention = "nettoyage"
        action = "détection des doublons"
        method = "duplicated().sum()"
    elif any(w in query_lower for w in ["outlier", "anomalie", "aberrant", "extreme", "extrême"]):
        intention = "nettoyage"
        action = "détection des outliers"
        method = "IQR method"
    
    # Exploration
    elif any(w in query_lower for w in ["type", "dtypes", "datatype", "structure"]):
        intention = "exploration"
        action = "types de données"
        method = "dtypes"
    elif any(w in query_lower for w in ["description", "resume", "résumé", "apercu", "overview"]):
        intention = "exploration"
        action = "description statistique"
        method = "describe()"
    elif any(w in query_lower for w in ["correlation", "corrélations", "matrice", "heatmap"]):
        intention = "exploration"
        action = "matrice de corrélation"
        method = "corr()"
    
    # Si pas de colonnes détectées mais intention spécifique
    if not target_cols:
        # Prendre première colonne numérique par défaut pour stats/viz
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols and intention in ["statistique", "visualisation"]:
            target_cols = [num_cols[0]]
        else:
            target_cols = ["toutes"]
    
    # Construction du plan
    plan = {
        "intention": intention,
        "action": action,
        "target_columns": target_cols,
        "method": method,
        "groupby": groupby_col,
        "confidence": "Haute" if len(target_cols) > 0 and target_cols[0] != "toutes" else "Moyenne",
        "explanation": f"Analyse '{action}'"
    }
    
    if groupby_col:
        plan["explanation"] += f" groupée par '{groupby_col}'"
    if target_cols and target_cols[0] != "toutes":
        plan["explanation"] += f" sur {', '.join(target_cols)}"
    else:
        plan["explanation"] += " sur l'ensemble des données"
    
    return plan

def generate_code(plan, df):
    """Génère le code Python selon le plan"""
    code_lines = []
    intention = plan['intention']
    cols = plan['target_columns']
    groupby = plan.get('groupby')
    
    # Import commun
    if intention == "visualisation":
        code_lines.append("import plotly.express as px")
        code_lines.append("import plotly.graph_objects as go")
    
    # Construction du code selon intention
    if intention == "statistique":
        if cols[0] != "toutes" and len(cols) == 1:
            col = cols[0]
            if groupby:
                code_lines.append(f"result = df.groupby('{groupby}')['{col}'].{plan['method'].replace('()', '')}().reset_index()")
                code_lines.append(f"result.columns = ['{groupby}', '{plan['method'].replace('()', '')}_{col}']")
            else:
                code_lines.append(f"result = df['{col}'].{plan['method'].replace('()', '')}()")
        else:
            if groupby:
                code_lines.append(f"result = df.groupby('{groupby}').{plan['method'].replace('()', '')}()")
            else:
                code_lines.append(f"result = df.{plan['method'].replace('()', '')}()")
    
    elif intention == "visualisation":
        col = cols[0] if cols[0] != "toutes" else df.select_dtypes(include=[np.number]).columns[0]
        
        if "histogram" in plan['method']:
            code_lines.append(f"fig = px.histogram(df, x='{col}', title='Distribution de {col}')")
            code_lines.append("st.plotly_chart(fig, use_container_width=True)")
            code_lines.append("result = 'Histogramme affiché'")
            
        elif "bar" in plan['method']:
            if groupby:
                agg_col = df.select_dtypes(include=[np.number]).columns[0]
                code_lines.append(f"agg_df = df.groupby('{groupby}')['{agg_col}'].sum().reset_index()")
                code_lines.append(f"fig = px.bar(agg_df, x='{groupby}', y='{agg_col}', title='{agg_col} par {groupby}')")
            else:
                code_lines.append(f"value_counts = df['{col}'].value_counts().head(20).reset_index()")
                code_lines.append(f"value_counts.columns = ['{col}', 'count']")
                code_lines.append(f"fig = px.bar(value_counts, x='{col}', y='count', title='Top 20 {col}')")
            code_lines.append("st.plotly_chart(fig, use_container_width=True)")
            code_lines.append("result = 'Diagramme en barres affiché'")
            
        elif "pie" in plan['method']:
            code_lines.append(f"value_counts = df['{col}'].value_counts().head(10).reset_index()")
            code_lines.append(f"value_counts.columns = ['{col}', 'count']")
            code_lines.append(f"fig = px.pie(value_counts, values='count', names='{col}', title='Répartition {col}')")
            code_lines.append("st.plotly_chart(fig, use_container_width=True)")
            code_lines.append("result = 'Camembert affiché'")
            
        elif "scatter" in plan['method']:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                x_col = cols[0] if cols[0] in num_cols else num_cols[0]
                y_col = cols[1] if len(cols) > 1 and cols[1] in num_cols else num_cols[1] if len(num_cols) > 1 else num_cols[0]
                color_col = f", color='{groupby}'" if groupby else ""
                code_lines.append(f"fig = px.scatter(df, x='{x_col}', y='{y_col}'{color_col}, title='{x_col} vs {y_col}')")
                code_lines.append("st.plotly_chart(fig, use_container_width=True)")
                code_lines.append("result = 'Nuage de points affiché'")
            else:
                code_lines.append("result = 'Besoin de 2 colonnes numériques pour un scatter plot'")
                
        elif "line" in plan['method']:
            if groupby:
                num_col = df.select_dtypes(include=[np.number]).columns[0]
                code_lines.append(f"agg_df = df.groupby('{groupby}')['{num_col}'].sum().reset_index()")
                code_lines.append(f"fig = px.line(agg_df, x='{groupby}', y='{num_col}', title='Évolution par {groupby}')")
            else:
                code_lines.append(f"fig = px.line(df, y='{col}', title='Évolution de {col}')")
            code_lines.append("st.plotly_chart(fig, use_container_width=True)")
            code_lines.append("result = 'Graphique linéaire affiché'")
    
    elif intention == "nettoyage":
        if "isnull" in plan['method']:
            code_lines.append("result = df.isnull().sum()")
        elif "duplicated" in plan['method']:
            code_lines.append("result = df.duplicated().sum()")
        elif "IQR" in plan['method']:
            num_col = df.select_dtypes(include=[np.number]).columns[0]
            code_lines.append(f"Q1 = df['{num_col}'].quantile(0.25)")
            code_lines.append(f"Q3 = df['{num_col}'].quantile(0.75)")
            code_lines.append(f"IQR = Q3 - Q1")
            code_lines.append(f"outliers = df[(df['{num_col}'] < Q1 - 1.5*IQR) | (df['{num_col}'] > Q3 + 1.5*IQR)]")
            code_lines.append("result = f\"{len(outliers)} outliers détectés\"")
    
    elif intention == "exploration":
        if "dtypes" in plan['method']:
            code_lines.append("result = df.dtypes")
        elif "describe" in plan['method']:
            code_lines.append("result = df.describe(include='all').transpose()")
        elif "corr" in plan['method']:
            num_df = "df.select_dtypes(include=[np.number])"
            code_lines.append(f"corr_matrix = {num_df}.corr()")
            code_lines.append("fig = px.imshow(corr_matrix, text_auto=True, aspect='auto', title='Matrice de corrélation')")
            code_lines.append("st.plotly_chart(fig, use_container_width=True)")
            code_lines.append("result = corr_matrix")
    
    else:
        code_lines.append("result = df.head(10)")
    
    return "\n".join(code_lines)

def execute_code_safe(code, df):
    """Exécute le code généré en toute sécurité"""
    local_vars = {
        'df': df.copy(),
        'pd': pd,
        'np': np,
        'st': st
    }
    
    try:
        exec(code, {"__builtins__": {}}, local_vars)
        result = local_vars.get('result', None)
        fig = local_vars.get('fig', None)
        return result, fig
    except Exception as e:
        raise Exception(f"Erreur d'exécution : {str(e)}\nCode :\n{code}")

# ==================== INTERFACE ====================

st.title("🧠 Data Brain Assistant")
st.markdown("*Analyse universelle de données — Tout fichier, toute question*")

# Upload fichier universel
st.header("1. Chargez votre fichier")
uploaded_file = st.file_uploader(
    "CSV, Excel, JSON, Parquet, ou texte structuré", 
    type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'txt']
)

if uploaded_file:
    try:
        file_name = uploaded_file.name.lower()
        
        # Détection intelligente du format
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith('.json'):
            try:
                df = pd.read_json(uploaded_file)
            except:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('utf-8')
                try:
                    data = json.loads(content)
                    df = pd.json_normalize(data) if isinstance(data, dict) else pd.DataFrame(data)
                except:
                    uploaded_file.seek(0)
                    data = [json.loads(line) for line in uploaded_file]
                    df = pd.DataFrame(data)
        elif file_name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        elif file_name.endswith('.txt'):
            uploaded_file.seek(0)
            content = uploaded_file.read().decode('utf-8')
            try:
                df = pd.read_csv(StringIO(content), sep=None, engine='python')
            except:
                df = pd.DataFrame({'content': content.split('\n')})
        else:
            st.error("Format non reconnu")
            st.stop()
        
        # Inférence automatique des types
        df = infer_types(df)
        st.session_state.data = df
        
        # Métadonnées
        st.success(f"✅ {df.shape[0]:,} lignes × {df.shape[1]} colonnes | Mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        with st.expander("🔍 Aperçu et structure"):
            tab1, tab2, tab3 = st.tabs(["Données", "Types", "Statistiques rapides"])
            
            with tab1:
                st.dataframe(df.head(10), use_container_width=True)
            
            with tab2:
                type_info = pd.DataFrame({
                    'Colonne': df.columns,
                    'Type détecté': df.dtypes.astype(str),
                    'Type original': [str(type(df[col].iloc[0])) if len(df) > 0 else 'N/A' for col in df.columns],
                    'Valeurs uniques': [df[col].nunique() for col in df.columns],
                    'Valeurs manquantes': [df[col].isnull().sum() for col in df.columns],
                    'Exemple': [str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else 'N/A' for col in df.columns]
                })
                st.dataframe(type_info, use_container_width=True)
            
            with tab3:
                st.write(df.describe(include='all').transpose())
                
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {str(e)}")
        st.info("💡 Essayez de vérifier l'encodage ou le format du fichier")

# Requête utilisateur
if st.session_state.data is not None:
    st.header("2. Posez votre question")
    
    # Suggestions contextuelles
    df = st.session_state.data
    suggestions = generate_suggestions(df)
    
    col_sugg1, col_sugg2, col_sugg3 = st.columns(3)
    with col_sugg1:
        if st.button(f"📊 {suggestions[0][:40]}...", use_container_width=True):
            st.session_state.suggestion = suggestions[0]
    with col_sugg2:
        if st.button(f"📈 {suggestions[1][:40]}...", use_container_width=True):
            st.session_state.suggestion = suggestions[1]
    with col_sugg3:
        if st.button(f"🔍 {suggestions[2][:40]}...", use_container_width=True):
            st.session_state.suggestion = suggestions[2]
    
    # Input avec suggestion pré-remplie
    default_query = st.session_state.get('suggestion', '')
    query = st.text_input(
        "Exemples : 'moyenne par région', 'histogramme des prix', 'corrélation entre age et revenu'",
        value=default_query,
        placeholder="Décrivez ce que vous voulez analyser..."
    )
    
    if query:
        with st.spinner("🧠 Analyse de la requête..."):
            # Parsing intelligent
            plan = parse_query_v2(query, df)
            
            # Affichage du plan
            st.subheader("📋 Plan d'action compris")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Intention :** `{plan['intention']}`")
                st.write(f"**Action :** {plan['action']}")
            with col2:
                st.write(f"**Colonnes :** {', '.join(plan['target_columns'])}")
                st.write(f"**Groupe :** {plan.get('groupby', 'Aucun')}")
            with col3:
                st.write(f"**Méthode :** `{plan['method']}`")
                st.write(f"**Confiance :** {plan.get('confidence', 'Moyenne')}")
            
            st.info(f"💡 {plan['explanation']}")
            
            # Génération du code
            st.subheader("🐍 Code généré")
            code_python = generate_code(plan, df)
            st.code(code_python, language='python')
            
            # Exécution
            st.subheader("🚀 Résultat")
            try:
                result, fig = execute_code_safe(code_python, df)
                
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result, use_container_width=True)
                        if len(result) > 100:
                            st.download_button(
                                "📥 Télécharger les résultats (CSV)",
                                result.to_csv(index=False).encode('utf-8'),
                                "resultats.csv",
                                "text/csv"
                            )
                    elif isinstance(result, (int, float)):
                        st.metric("Résultat", f"{result:,.2f}" if isinstance(result, float) else f"{result:,}")
                    elif isinstance(result, str):
                        st.success(result)
                    else:
                        st.write(result)
                
                # Sauvegarde historique
                st.session_state.history.append({
                    'timestamp': pd.Timestamp.now().strftime("%H:%M:%S"),
                    'query': query,
                    'intention': plan['intention'],
                    'success': True
                })
                
                # Limite historique à 20
                if len(st.session_state.history) > 20:
                    st.session_state.history = st.session_state.history[-20:]
                
            except Exception as e:
                st.error(f"❌ Erreur d'exécution : {str(e)}")
                st.info("💡 Essayez de reformuler votre question ou vérifiez les colonnes concernées")
                
                st.session_state.history.append({
                    'timestamp': pd.Timestamp.now().strftime("%H:%M:%S"),
                    'query': query,
                    'intention': plan['intention'],
                    'success': False,
                    'error': str(e)
                })

# Sidebar historique et infos
with st.sidebar:
    st.header("📜 Historique des requêtes")
    
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-10:])):
            icon = "✅" if item['success'] else "❌"
            st.write(f"{icon} **{item['timestamp']}** | {item['query'][:30]}...")
            st.caption(f"Intention: {item['intention']}")
            st.divider()
    else:
        st.write("Aucune requête encore")
    
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.history = []
        st.rerun()
    
    st.divider()
    st.header("ℹ️ Aide")
    st.markdown("""
    **Types de requêtes supportés :**
    - Statistiques : moyenne, médiane, écart-type, min, max
    - Visualisations : histogramme, scatter, ligne, barres, camembert
    - Nettoyage : détection manquants, doublons, outliers
    - Exploration : describe, types, corrélations
    - Groupby : analyses par catégorie
    
    **Astuce :** Utilisez "par [colonne]" pour grouper les résultats.
    """)
