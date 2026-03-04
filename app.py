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
    """Génère du code Python robuste et universel"""
    code_lines = []
    intention = plan['intention']
    cols = plan['target_columns']
    groupby = plan.get('groupby')
    
    # Import conditionnel
    if intention == "visualisation":
        code_lines.append("import plotly.express as px")
        code_lines.append("import time")  # Pour les keys uniques
    
    # Fonction helper pour obtenir colonnes numériques
    code_lines.append("# Détection automatique des types")
    code_lines.append("num_cols = df.select_dtypes(include=['number']).columns.tolist()")
    code_lines.append("cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()")
    
    if intention == "statistique":
        target_col = cols[0] if cols[0] != "toutes" else "num_cols[0] if num_cols else df.columns[0]"
        
        if target_col != "num_cols[0] if num_cols else df.columns[0]":
            # Colonne spécifique
            if groupby:
                code_lines.append(f"if '{target_col}' in num_cols:")
                code_lines.append(f"    result = df.groupby('{groupby}')['{target_col}'].agg(['mean', 'sum', 'count', 'min', 'max']).reset_index()")
                code_lines.append(f"    result = result.rename(columns={{'mean': 'moyenne_{target_col}', 'sum': 'total_{target_col}'}})")
                code_lines.append("else:")
                code_lines.append(f"    result = 'Erreur: {target_col} n\\'est pas numérique'")
            else:
                code_lines.append(f"if '{target_col}' in num_cols:")
                code_lines.append(f"    result = df['{target_col}'].agg(['mean', 'std', 'min', 'max', 'median'])")
                code_lines.append("else:")
                code_lines.append(f"    result = df['{target_col}'].describe() if '{target_col}' in cat_cols else 'Colonne non trouvée'")
        else:
            # Toutes les colonnes numériques
            if groupby:
                code_lines.append(f"if '{groupby}' in cat_cols and num_cols:")
                code_lines.append(f"    result = df.groupby('{groupby}')[num_cols].mean().reset_index()")
                code_lines.append("else:")
                code_lines.append("    result = 'Groupby impossible: vérifiez les types'")
            else:
                code_lines.append("result = df[num_cols].describe() if num_cols else 'Aucune colonne numérique'")
    
    elif intention == "visualisation":
        col = cols[0] if cols[0] != "toutes" else "num_cols[0] if num_cols else df.columns[0]"
        
        if "histogram" in plan['method']:
            code_lines.append(f"target = '{col}' if '{col}' in df.columns else (num_cols[0] if num_cols else df.columns[0])")
            code_lines.append("if target in num_cols:")
            code_lines.append("    fig = px.histogram(df, x=target, title=f'Distribution de {target}')")
            code_lines.append("    unique_key = f'viz_{int(time.time()*1000) % 1000000}'")  # Généré DANS le code
            code_lines.append("    st.plotly_chart(fig, use_container_width=True, key=unique_key)")
            code_lines.append("    result = f'Histogramme de {target}'")
            code_lines.append("else:")
            code_lines.append("    result = f'{target} n\\'est pas numérique, histogramme impossible'")
            
        elif "bar" in plan['method']:
            if groupby:
                code_lines.append(f"agg_col = '{cols[1]}' if len({cols}) > 1 and '{cols[1]}' in num_cols else (num_cols[0] if num_cols else None)")
                code_lines.append("if agg_col:")
                code_lines.append(f"    agg_df = df.groupby('{groupby}')[agg_col].sum().reset_index().sort_values(agg_col, ascending=False).head(20)")
                code_lines.append("    fig = px.bar(agg_df, x=groupby, y=agg_col, title=f'Total par {groupby}')")
                code_lines.append("    unique_key = f'viz_{int(time.time()*1000) % 1000000}'")
                code_lines.append("    st.plotly_chart(fig, use_container_width=True, key=unique_key)")
                code_lines.append("    result = agg_df")
                code_lines.append("else:")
                code_lines.append("    result = 'Pas de colonne numérique pour le diagramme en barres'")
            else:
                code_lines.append(f"if '{col}' in cat_cols:")
                code_lines.append(f"    value_counts = df['{col}'].value_counts().head(15).reset_index()")
                code_lines.append(f"    value_counts.columns = ['{col}', 'count']")
                code_lines.append(f"    fig = px.bar(value_counts, x='{col}', y='count', title='Top 15 {col}')")
                code_lines.append("    unique_key = f'viz_{int(time.time()*1000) % 1000000}'")
                code_lines.append("    st.plotly_chart(fig, use_container_width=True, key=unique_key)")
                code_lines.append("    result = value_counts")
                code_lines.append("else:")
                code_lines.append(f"    result = '{col} n\\'est pas catégoriel'")
        
        elif "scatter" in plan['method']:
            code_lines.append("if len(num_cols) >= 2:")
            code_lines.append(f"    x_col = '{cols[0]}' if '{cols[0]}' in num_cols else num_cols[0]")
            y_col = cols[1] if len(cols) > 1 else "num_cols[1] if len(num_cols) > 1 else num_cols[0]"
            code_lines.append(f"    y_col = '{cols[1]}' if len(['{cols[0]}']) > 1 and '{cols[1]}' in num_cols else {y_col}")
            color_code = f", color='{groupby}'" if groupby else ""
            code_lines.append(f"    fig = px.scatter(df, x=x_col, y=y_col{color_code}, title=f'{{x_col}} vs {{y_col}}')")
            code_lines.append("    unique_key = f'viz_{int(time.time()*1000) % 1000000}'")
            code_lines.append("    st.plotly_chart(fig, use_container_width=True, key=unique_key)")
            code_lines.append("    result = f'Scatter: {{x_col}} vs {{y_col}}'")
            code_lines.append("else:")
            code_lines.append("    result = 'Besoin de 2 colonnes numériques'")
        
        elif "pie" in plan['method']:
            code_lines.append(f"if '{col}' in cat_cols:")
            code_lines.append(f"    value_counts = df['{col}'].value_counts().head(10).reset_index()")
            code_lines.append(f"    value_counts.columns = ['{col}', 'count']")
            code_lines.append(f"    fig = px.pie(value_counts, values='count', names='{col}', title='Répartition {col}')")
            code_lines.append("    unique_key = f'viz_{int(time.time()*1000) % 1000000}'")
            code_lines.append("    st.plotly_chart(fig, use_container_width=True, key=unique_key)")
            code_lines.append("    result = value_counts")
            code_lines.append("else:")
            code_lines.append(f"    result = '{col} n\\'est pas catégoriel'")
        
        else:  # line par défaut
            code_lines.append("if num_cols:")
            code_lines.append(f"    y_col = '{col}' if '{col}' in num_cols else num_cols[0]")
            code_lines.append("    fig = px.line(df, y=y_col, title=f'Évolution de {y_col}')")
            code_lines.append("    unique_key = f'viz_{int(time.time()*1000) % 1000000}'")
            code_lines.append("    st.plotly_chart(fig, use_container_width=True, key=unique_key)")
            code_lines.append("    result = f'Graphique linéaire de {y_col}'")
            code_lines.append("else:")
            code_lines.append("    result = 'Pas de colonne numérique'")
    
    elif intention == "nettoyage":
        if "isnull" in plan['method']:
            code_lines.append("result = df.isnull().sum().sort_values(ascending=False)")
            code_lines.append("result = result[result > 0]")
            code_lines.append("if result.empty:")
            code_lines.append("    result = 'Aucune valeur manquante'")
        elif "duplicated" in plan['method']:
            code_lines.append("dup_count = df.duplicated().sum()")
            code_lines.append("result = f'{dup_count} lignes dupliquées ({dup_count/len(df)*100:.1f}%)'")
        elif "IQR" in plan['method']:
            code_lines.append("if num_cols:")
            code_lines.append("    col = num_cols[0]")
            code_lines.append("    Q1 = df[col].quantile(0.25)")
            code_lines.append("    Q3 = df[col].quantile(0.75)")
            code_lines.append("    IQR = Q3 - Q1")
            code_lines.append("    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]")
            code_lines.append("    result = f'{len(outliers)} outliers sur {col} (IQR method)'")
            code_lines.append("else:")
            code_lines.append("    result = 'Pas de colonne numérique'")
    
    elif intention == "exploration":
        if "dtypes" in plan['method']:
            code_lines.append("result = pd.DataFrame({'Type': df.dtypes, 'Non_Null': df.count(), 'Null': df.isnull().sum(), 'Unique': df.nunique()})")
        elif "describe" in plan['method']:
            code_lines.append("result = df.describe(include='all').transpose()")
        elif "corr" in plan['method']:
            code_lines.append("if len(num_cols) > 1:")
            code_lines.append("    corr_matrix = df[num_cols].corr()")
            code_lines.append("    fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto', title='Matrice de corrélation')")
            code_lines.append("    unique_key = f'viz_{int(time.time()*1000) % 1000000}'")
            code_lines.append("    st.plotly_chart(fig, use_container_width=True, key=unique_key)")
            code_lines.append("    result = corr_matrix")
            code_lines.append("else:")
            code_lines.append("    result = 'Besoin de 2+ colonnes numériques'")
    else:
        code_lines.append("result = df.head(10)")
    
    return "\n".join(code_lines)

def execute_code_safe(code, df):
    """Exécute le code en autorisant les imports nécessaires"""
    
    # Builtins minimaux pour faire fonctionner le code
    safe_builtins = {
        '__import__': __import__,  # AUTORISE les imports
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'str': str,
        'int': int,
        'float': float,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'type': type,
        'isinstance': isinstance,
        'hasattr': hasattr,
        'getattr': getattr,
        'print': print,  # Pour debug
    }
    
    local_vars = {
        'df': df.copy(),
        'pd': pd,
        'np': np,
        'st': st,
    }
    
    try:
        exec(code, {"__builtins__": safe_builtins}, local_vars)
        result = local_vars.get('result', None)
        fig = local_vars.get('fig', None)
        return result, fig
    except Exception as e:
        raise Exception(f"Exécution: {str(e)}")

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
