import streamlit as st
import pandas as pd
import json
import re

st.set_page_config(page_title="Data Brain Assistant", layout="wide")

# Initialisation session state (OBLIGATOIRE en premier)
if 'data' not in st.session_state:
    st.session_state.data = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'plan' not in st.session_state:
    st.session_state.plan = None

st.title("🧠 Data Brain Assistant")
st.markdown("*Analyse intelligente sans API externe*")

# Upload fichier
st.header("1. Chargez votre fichier")
uploaded_file = st.file_uploader("CSV, Excel ou JSON", type=['csv', 'xlsx', 'xls', 'json'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            try:
                df = pd.read_json(uploaded_file)
            except:
                uploaded_file.seek(0)
                data = [json.loads(line) for line in uploaded_file]
                df = pd.DataFrame(data)
        
        st.session_state.data = df
        st.success(f"✅ {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        with st.expander("Voir les données"):
            st.dataframe(df.head())
            st.write("**Colonnes disponibles :**")
            for i, col in enumerate(df.columns):
                st.write(f"{i+1}. `{col}` ({df[col].dtype})")
                
    except Exception as e:
        st.error(f"Erreur : {e}")

# Fonction de parsing intelligent (sans API)
def parse_query(query, columns):
    query_lower = query.lower()
    detected_cols = []
    
    # Détection des colonnes mentionnées
    for col in columns:
        if col.lower() in query_lower:
            detected_cols.append(col)
    
    # Si aucune colonne détectée exactement, recherche floue
    if not detected_cols:
        words = query_lower.split()
        for word in words:
            for col in columns:
                if word in col.lower() or col.lower() in word:
                    if col not in detected_cols:
                        detected_cols.append(col)
    
    # Détection de l'intention
    intention = "exploration"
    action = "analyse générale"
    methode = "describe()"
    
    # Statistiques
    if any(w in query_lower for w in ["moyenne", "mean", "moyen", "average"]):
        intention = "statistique"
        action = "calcul de la moyenne"
        methode = "mean()"
    elif any(w in query_lower for w in ["somme", "total", "sum"]):
        intention = "statistique"
        action = "calcul de la somme"
        methode = "sum()"
    elif any(w in query_lower for w in ["minimum", "min", "plus petit"]):
        intention = "statistique"
        action = "valeur minimale"
        methode = "min()"
    elif any(w in query_lower for w in ["maximum", "max", "plus grand"]):
        intention = "statistique"
        action = "valeur maximale"
        methode = "max()"
    elif any(w in query_lower for w in ["mediane", "median", "médiane"]):
        intention = "statistique"
        action = "calcul de la médiane"
        methode = "median()"
    elif any(w in query_lower for w in ["ecart-type", "std", "standard", "variance"]):
        intention = "statistique"
        action = "calcul de l'écart-type"
        methode = "std()"
    
    # Visualisation
    elif any(w in query_lower for w in ["histogramme", "hist", "distribution"]):
        intention = "visualisation"
        action = "histogramme de distribution"
        methode = "hist() / plotly"
    elif any(w in query_lower for w in ["graphique", "plot", "courbe", "courbe", "evolution"]):
        intention = "visualisation"
        action = "graphique linéaire"
        methode = "line plot"
    elif any(w in query_lower for w in ["nuage", "scatter", "correlation", "corrélation"]):
        intention = "visualisation"
        action = "nuage de points"
        methode = "scatter plot"
    
    # Nettoyage
    elif any(w in query_lower for w in ["manquant", "missing", "null", "na", "vide"]):
        intention = "nettoyage"
        action = "détection des valeurs manquantes"
        methode = "isnull().sum()"
    elif any(w in query_lower for w in ["doublon", "duplicate", "dupli"]):
        intention = "nettoyage"
        action = "détection des doublons"
        methode = "duplicated()"
    
    # Exploration
    elif any(w in query_lower for w in ["type", "dtypes", "datatype"]):
        intention = "exploration"
        action = "types de données"
        methode = "dtypes"
    elif any(w in query_lower for w in ["description", "resume", "résumé", "apercu"]):
        intention = "exploration"
        action = "description statistique"
        methode = "describe()"
    
    # Si colonne détectée mais pas d'intention spécifique
    if detected_cols and intention == "exploration":
        action = f"analyse de {detected_cols[0]}"
    
    return {
        "intention": intention,
        "action": action,
        "colonnes_concernees": detected_cols if detected_cols else ["toutes"],
        "methode": methode,
        "explication": f"J'ai détecté une demande d'{action} sur {', '.join(detected_cols) if detected_cols else 'les données'}."
    }

# Requête utilisateur
if st.session_state.data is not None:
    st.header("2. Posez votre question")
    
    query = st.text_input(
        "Exemples : 'moyenne de Units_Sold', 'histogramme des prix', 'valeurs manquantes'",
        placeholder="Votre question ici..."
    )
    
    if query:
        with st.spinner("🧠 Analyse de la requête..."):
            # Parsing local
            plan = parse_query(query, list(st.session_state.data.columns))
            
            # Affichage du plan
            st.subheader("📋 Plan d'action compris")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Intention :** `{plan['intention']}`")
                st.write(f"**Action :** {plan['action']}")
            with col2:
                st.write(f"**Colonnes :** {', '.join(plan['colonnes_concernees'])}")
                st.write(f"**Méthode :** `{plan['methode']}`")
            
            st.info(f"💡 {plan['explication']}")
            st.session_state['plan'] = plan
            
            # Génération du code Python
            st.subheader("🐍 Code généré")
            
            df = st.session_state.data
            code_lines = []
            
            if plan['intention'] == 'statistique':
                if plan['colonnes_concernees'][0] != 'toutes':
                    col = plan['colonnes_concernees'][0]
                    if 'moyenne' in plan['action']:
                        code_lines.append(f"result = df['{col}'].mean()")
                    elif 'somme' in plan['action']:
                        code_lines.append(f"result = df['{col}'].sum()")
                    elif 'min' in plan['action']:
                        code_lines.append(f"result = df['{col}'].min()")
                    elif 'max' in plan['action']:
                        code_lines.append(f"result = df['{col}'].max()")
                    elif 'mediane' in plan['action']:
                        code_lines.append(f"result = df['{col}'].median()")
                    elif 'ecart-type' in plan['action']:
                        code_lines.append(f"result = df['{col}'].std()")
                else:
                    code_lines.append("result = df.describe()")
                    
            elif plan['intention'] == 'nettoyage':
                if 'manquant' in plan['action']:
                    code_lines.append("result = df.isnull().sum()")
                elif 'doublon' in plan['action']:
                    code_lines.append("result = df.duplicated().sum()")
                    
            elif plan['intention'] == 'exploration':
                if 'types' in plan['action']:
                    code_lines.append("result = df.dtypes")
                else:
                    code_lines.append("result = df.describe()")
                    
            elif plan['intention'] == 'visualisation':
                col = plan['colonnes_concernees'][0] if plan['colonnes_concernees'][0] != 'toutes' else df.columns[0]
                if 'histogramme' in plan['action']:
                    code_lines.append(f"import plotly.express as px")
                    code_lines.append(f"fig = px.histogram(df, x='{col}', title='Distribution de {col}')")
                    code_lines.append(f"fig.show()")
                    code_lines.append(f"result = 'Histogramme généré'")
                elif 'scatter' in plan['action'] or 'nuage' in plan['action']:
                    if len(plan['colonnes_concernees']) >= 2:
                        col2 = plan['colonnes_concernees'][1]
                        code_lines.append(f"import plotly.express as px")
                        code_lines.append(f"fig = px.scatter(df, x='{col}', y='{col2}', title='{col} vs {col2}')")
                        code_lines.append(f"fig.show()")
                        code_lines.append(f"result = 'Nuage de points généré'")
                    else:
                        code_lines.append(f"result = 'Besoin de 2 colonnes pour un scatter plot'")
                else:
                    code_lines.append(f"import plotly.express as px")
                    code_lines.append(f"fig = px.line(df, y='{col}', title='Évolution de {col}')")
                    code_lines.append(f"fig.show()")
                    code_lines.append(f"result = 'Graphique généré'")
            
            # Affichage du code
            code_python = "\n".join(code_lines)
            st.code(code_python, language='python')
            
            # Exécution
            st.subheader("🚀 Résultat")
            try:
                local_vars = {'df': df, 'pd': pd, 'st': st}
                
                if 'plotly' in code_python:
                    import plotly.express as px
                    local_vars['px'] = px
                
                exec(code_python, local_vars)
                
                if 'result' in local_vars:
                    result = local_vars['result']
                    if isinstance(result, (int, float)):
                        st.metric("Résultat", f"{result:,.2f}")
                    elif isinstance(result, pd.Series):
                        st.write(result)
                    elif isinstance(result, str):
                        st.success(result)
                    else:
                        st.write(result)
                        
                # Sauvegarde historique
                st.session_state.history.append({
                    'query': query,
                    'plan': plan,
                    'result': 'success'
                })
                
            except Exception as e:
                st.error(f"❌ Erreur d'exécution : {e}")
                st.info("💡 Essayez de reformuler votre question")

# Historique
if st.session_state.history:
    with st.sidebar:
        st.header("📜 Historique")
        for i, item in enumerate(st.session_state.history[-5:]):
            st.write(f"{i+1}. {item['query'][:30]}...")
