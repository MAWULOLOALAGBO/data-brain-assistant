import streamlit as st
import pandas as pd
import json
import requests

st.set_page_config(page_title="Data Brain Assistant", layout="wide")

if 'data' not in st.session_state:
    st.session_state.data = None

st.title("🧠 Data Brain Assistant")

# Sidebar Hugging Face
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Token Hugging Face", type="password")
    if api_key:
        st.success("✅ Connecté à Hugging Face")
    else:
        st.warning("Entrez votre token HF")

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
        "Exemple : 'Quel est le prix moyen ?'",
        placeholder="Votre question ici..."
    )
    
    if query:
        with st.spinner("🧠 Analyse..."):
            df = st.session_state.data
            context = {
                "colonnes": list(df.columns),
                "types": {col: str(df[col].dtype) for col in df.columns},
                "exemple": df.head(2).to_dict()
            }
            
            prompt = f"""Tu es un assistant d'analyse de données. 
            
Contexte :
{json.dumps(context, indent=2, default=str)}

Question : "{query}"

Réponds UNIQUEMENT avec ce JSON :
{{
    "intention": "statistique",
    "action": "description",
    "colonnes_concernees": ["colonne"],
    "methode": "méthode",
    "explication": "explication"
}}

JSON :"""
            
            # Appel Hugging Face (modèle Mistral gratuit)
            try:
                response = requests.post(
                    url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 300,
                            "temperature": 0.1,
                            "return_full_text": False
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
                    
                    # Extraction JSON
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    
                    # Trouve le JSON dans le texte
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_str = content[start:end]
                    
                    plan = json.loads(json_str)
                    
                    # Affichage
                    st.subheader("📋 Plan d'action")
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
                    
                else:
                    st.error(f"Erreur API : {response.status_code}")
                    st.write(response.text)
                    
            except Exception as e:
                st.error(f"Erreur : {e}")
