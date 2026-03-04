import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Data Brain Assistant", layout="wide")

st.title("🧠 Data Brain Assistant")
st.markdown("*Chargez n'importe quel fichier pour commencer*")

# Upload
uploaded_file = st.file_uploader(
    "Déposez votre fichier", 
    type=['csv', 'xlsx', 'xls', 'json']
)

if uploaded_file:
    # Détection du format
    file_name = uploaded_file.name
    st.write(f"📁 Fichier détecté : `{file_name}`")
    
    try:
        # Chargement selon l'extension
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV chargé")
            
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            st.success("✅ Excel chargé")
            
        elif file_name.endswith('.json'):
            # Essai JSON normal
            try:
                df = pd.read_json(uploaded_file)
            except:
                # Essai JSON lignes (JSONL)
                uploaded_file.seek(0)
                data = [json.loads(line) for line in uploaded_file]
                df = pd.DataFrame(data)
            st.success("✅ JSON chargé")
        
        # Affichage des métadonnées
        st.subheader("📊 Informations du fichier")
        col1, col2, col3 = st.columns(3)
        col1.metric("Lignes", df.shape[0])
        col2.metric("Colonnes", df.shape[1])
        col3.metric("Taille mémoire", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Aperçu
        st.subheader("👁️ Aperçu (5 premières lignes)")
        st.dataframe(df.head())
        
        # Liste des colonnes
        st.subheader("📋 Colonnes détectées")
        for col in df.columns:
            st.write(f"- **{col}** : {df[col].dtype}")
            
        # Stockage pour étapes futures
        st.session_state['data'] = df
        st.session_state['file_name'] = file_name
        
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {str(e)}")
        st.info("💡 Vérifiez que le fichier n'est pas corrompu")
