import streamlit as st
import pandas as pd
import json
import openai

st.set_page_config(page_title="Data Brain Assistant", layout="wide")

# Initialisation
if 'data' not in st.session_state:
    st.session_state.data = None

st.title("🧠 Data Brain Assistant")

# Sidebar pour la clé API
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Clé API OpenAI", type="password")
    if api_key:
        openai.api_key = api_key
        st.success("✅ Connecté")
    else:
        st.warning("Entrez votre clé API pour continuer")

# Upload fichier
st.header("1. Chargez votre fichier")
uploaded_file = st.file_uploader("CSV, Excel ou JSON", type=['csv', 'xlsx', 'xls', 'json'])

if uploaded_file:
    # Chargement (même code qu'avant)
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
        
        # Infos
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
            # Préparation du contexte
            df = st.session_state.data
            context = {
                "colonnes": list(df.columns),
                "types": {col: str(df[col].dtype) for col in df.columns},
                "exemple": df.head(2).to_dict()
            }
            
            # Prompt GPT
            prompt = f"""Tu es un assistant d'analyse de données. 
            
Contexte du fichier :
{json.dumps(context, indent=2)}

Requête utilisateur : "{query}"

Analyse cette requête et réponds UNIQUEMENT avec ce JSON :
{{
    "intention": "statistique|visualisation|nettoyage|prediction|exploration|inconnu",
    "action": "description de ce qu'il faut faire",
    "colonnes_concernees": ["colonne1", "colonne2"],
    "methode": "méthode technique",
    "explication": "explication simple pour l'utilisateur"
}}

JSON :"""
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                
                # Extraction JSON
                content = response.choices[0].message.content
                
                # Nettoyage si markdown
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                plan = json.loads(content.strip())
                
                # Affichage du plan
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
                st.success("✅ Requête comprise (exécution à l'étape suivante)")
                
            except Exception as e:
                st.error(f"Erreur GPT : {e}")
                st.code(content if 'content' in locals() else "Pas de réponse")
