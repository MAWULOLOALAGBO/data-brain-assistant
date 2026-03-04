import streamlit as st
import pandas as pd
import json
import requests

st.set_page_config(page_title="Data Brain Assistant", layout="wide")

# Initialisation
if 'data' not in st.session_state:
    st.session_state.data = None

st.title("🧠 Data Brain Assistant")

# Sidebar pour la clé API
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Clé API OpenRouter", type="password")
    if api_key:
        st.success("✅ Connecté à OpenRouter")
    else:
        st.warning("Entrez votre clé OpenRouter")

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
            st.write("**Colonnes :**", list(df.columns))
            
    except Exception as e:
        st.error(f"Erreur : {e}")

# Requête utilisateur
if st.session_state.data is not None and api_key:
    st.header("2. Posez votre question")
    
    query = st.text_input(
        "Exemple : 'Quel est le prix moyen ?' ou 'Histogramme des quantités'",
        placeholder="Votre question ici..."
    )
    
    if query:
        with st.spinner("🧠 Analyse de la requête..."):
            df = st.session_state.data
            context = {
                "colonnes": list(df.columns),
                "types": {col: str(df[col].dtype) for col in df.columns},
                "exemple": df.head(2).to_dict()
            }
            
            prompt = f"""Tu es un assistant d'analyse de données. 
            
Contexte du fichier :
{json.dumps(context, indent=2, default=str)}

Requête utilisateur : "{query}"

Analyse cette requête et réponds UNIQUEMENT avec ce JSON :
{{
    "intention": "statistique|visualisation|nettoyage|prediction|exploration",
    "action": "description de ce qu'il faut faire",
    "colonnes_concernees": ["colonne1", "colonne2"],
    "methode": "méthode technique",
    "explication": "explication simple pour l'utilisateur"
}}

JSON :"""
            
            # Appel OpenRouter (modèle Mistral gratuit)
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://data-brain-assistant.streamlit.app",
                        "X-Title": "Data Brain Assistant"
                    },
                    json={
                        "model": "meta-llama/llama-3.1-8b-instruct:free",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0
                    },
                    timeout=30
                )
                
                response_data = response.json()
                
                if 'error' in response_data:
                    st.error(f"Erreur API : {response_data['error']}")
                else:
                    content = response_data['choices'][0]['message']['content']
                    
                    # Nettoyage JSON
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    
                    plan = json.loads(content.strip())
                    
                    # Affichage
                    st.subheader("📋 Plan d'action compris")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Intention :** {plan['intention']}")
                        st.write(f"**Action :** {plan['action']}")
                    with col2:
                        st.write(f"**Colonnes :** {', '.join(plan['colonnes_concernees'])}")
                        st.write(f"**Méthode :** {plan['methode']}")
                    
                    st.info(f"💡 {plan['explication']}")
                    st.session_state['plan'] = plan
                    st.success("✅ Requête comprise")
                    
            except Exception as e:
                st.error(f"Erreur : {e}")
                if 'response_data' in locals():
                    st.write("Réponse brute :", response_data)
