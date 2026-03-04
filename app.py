import streamlit as st
import pandas as pd
import json
import re
from typing import Dict, List, Any

st.set_page_config(page_title="Data Brain Assistant", layout="wide")

# Initialisation
for key in ['data', 'history', 'plan', 'column_types', 'synonyms']:
    if key not in st.session_state:
        st.session_state[key] = {} if key in ['column_types', 'synonyms'] else ([] if key == 'history' else None)

st.title("🧠 Data Brain Assistant")
st.markdown("*Analyse universelle de n'importe quel fichier*")

# ==================== FONCTIONS UNIVERSELLES ====================

def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Détecte automatiquement le type sémantique de chaque colonne"""
    types = {}
    
    for col in df.columns:
        dtype = df[col].dtype
        sample = df[col].dropna().head(10).tolist()
        sample_str = ' '.join([str(s).lower() for s in sample])
        
        # Détection numérique
        if pd.api.types.is_numeric_dtype(dtype):
            # Est-ce un ID ?
            if 'id' in col.lower() or all(str(x).isdigit() and int(x) > 10000 for x in sample if pd.notna(x)):
                types[col] = 'id'
            # Est-ce une date (année) ?
            elif any(year in col.lower() for year in ['year', 'annee', 'mois', 'month', 'jour', 'day', 'date']):
                types[col] = 'temporal'
            # Est-ce un prix/valeur monétaire ?
            elif any(money in col.lower() for money in ['price', 'prix', 'cost', 'cout', 'revenue', 'montant', 'salary', 'salaire', 'eur', 'usd', '€', '$']):
                types[col] = 'monetary'
            # Est-ce une quantité ?
            elif any(qty in col.lower() for qty in ['quantity', 'quantite', 'count', 'nombre', 'units', 'total', 'nombre', 'volume']):
                types[col] = 'quantity'
            # Est-ce un pourcentage ?
            elif any(pct in col.lower() for pct in ['percentage', 'pourcent', 'rate', 'taux', 'ratio', 'share', 'part']):
                types[col] = 'percentage'
            else:
                types[col] = 'numeric'
                
        # Détection texte/catégoriel
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            unique_ratio = df[col].nunique() / len(df)
            unique_count = df[col].nunique()
            
            # Est-ce un genre/sexe ?
            if any(g in col.lower() for g in ['gender', 'sexe', 'sex']) or set(sample).issubset({'Male', 'Female', 'M', 'F', 'Homme', 'Femme', '0', '1', 'M', 'F'}):
                types[col] = 'gender'
            # Est-ce une localisation ?
            elif any(loc in col.lower() for loc in ['country', 'pays', 'region', 'city', 'ville', 'location', 'zone', 'area', 'departement', 'state']):
                types[col] = 'location'
            # Est-ce un type/catégorie ?
            elif any(cat in col.lower() for cat in ['type', 'category', 'categorie', 'class', 'classe', 'group', 'groupe', 'segment']):
                types[col] = 'category'
            # Est-ce un nom ?
            elif any(name in col.lower() for name in ['name', 'nom', 'title', 'titre', 'label']):
                types[col] = 'name'
            # Binaire (oui/non, true/false) ?
            elif set(sample).issubset({'Yes', 'No', 'Oui', 'Non', 'True', 'False', '0', '1', 'Y', 'N'}):
                types[col] = 'binary'
            # ID textuel ?
            elif 'id' in col.lower() or unique_ratio > 0.9:
                types[col] = 'id_text'
            # Catégorielle avec peu de valeurs uniques
            elif unique_count < 20 or unique_ratio < 0.1:
                types[col] = 'categorical'
            else:
                types[col] = 'text'
        
        # Dates
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            types[col] = 'datetime'
            
    return types

def find_column_by_semantic(query: str, columns: List[str], col_types: Dict[str, str]) -> List[str]:
    """Trouve les colonnes par sens, pas juste par nom exact"""
    query_lower = query.lower()
    matches = []
    
    # Mots-clés universels par type
    semantic_keywords = {
        'gender': ['genre', 'sexe', 'gender', 'sex', 'homme', 'femme', 'male', 'female', 'masculin', 'feminin'],
        'location': ['region', 'pays', 'country', 'ville', 'city', 'zone', 'area', 'lieu', 'place', 'localisation', 'geo'],
        'temporal': ['date', 'temps', 'time', 'annee', 'year', 'mois', 'month', 'jour', 'day', 'periode', 'period'],
        'monetary': ['prix', 'price', 'cout', 'cost', 'revenu', 'revenue', 'argent', 'money', 'montant', 'amount', 'salaire', 'salary', 'gain', 'profit'],
        'quantity': ['quantite', 'quantity', 'nombre', 'count', 'number', 'total', 'volume', 'unites', 'units', 'volume', 'somme'],
        'percentage': ['pourcentage', 'percentage', 'taux', 'rate', 'ratio', 'proportion', 'part', 'share'],
        'numeric': ['valeur', 'value', 'mesure', 'measure', 'score', 'points', 'indice', 'index']
    }
    
    # 1. Recherche par type sémantique
    for col, col_type in col_types.items():
        if col_type in semantic_keywords:
            keywords = semantic_keywords[col_type]
            if any(kw in query_lower for kw in keywords):
                if col not in matches:
                    matches.append(col)
    
    # 2. Recherche par nom de colonne (fuzzy)
    for col in columns:
        col_lower = col.lower().replace('_', ' ')
        # Mot exact ou partie du mot
        if col_lower in query_lower or any(word in col_lower for word in query_lower.split()):
            if col not in matches:
                matches.append(col)
        # Suppression des suffixes/prefixes communs
        clean_col = re.sub(r'(id_|_id|_name|_code|_num|number|n°|no)', '', col_lower)
        if clean_col in query_lower and len(clean_col) > 2:
            if col not in matches:
                matches.append(col)
    
    # 3. Recherche par contenu (valeurs uniques)
    if not matches:
        for col in columns:
            if col_types.get(col) in ['categorical', 'gender', 'location', 'binary']:
                try:
                    unique_vals = [str(v).lower() for v in st.session_state.data[col].dropna().unique()[:10]]
                    if any(val in query_lower for val in unique_vals if len(val) > 2):
                        matches.append(col)
                        break
                except:
                    pass
    
    return matches if matches else columns[:1]  # Retourne première colonne par défaut

def parse_query_universal(query: str, df: pd.DataFrame, col_types: Dict[str, str]) -> Dict[str, Any]:
    """Parseur universel qui comprend le sens, pas juste les mots"""
    query_lower = query.lower()
    columns = list(df.columns)
    
    # Détection des colonnes concernées (sémantique)
    detected_cols = find_column_by_semantic(query, columns, col_types)
    
    # Détection de l'action par patterns universels
    intention = "exploration"
    action = "analyse générale"
    method = "describe()"
    params = {}
    
    # PATTERNS STATISTIQUES
    stat_patterns = {
        'mean': [r'moyenne|mean|average|avg|moyen'],
        'sum': [r'somme|sum|total|aggregate'],
        'min': [r'minimum|min|plus petit|smallest|lowest'],
        'max': [r'maximum|max|plus grand|largest|biggest|highest'],
        'median': [r'mediane|median|milieu|middle'],
        'std': [r'ecart[- ]?type|std|standard|deviation|dispersion'],
        'count': [r'compte|count|nombre|how many|combien'],
        'unique': [r'unique|distinct|different|valeurs? uniques?']
    }
    
    for stat_name, patterns in stat_patterns.items():
        if any(re.search(p, query_lower) for p in patterns):
            intention = "statistique"
            action = f"calcul {stat_name}"
            method = f"{stat_name}()"
            params['stat'] = stat_name
            break
    
    # PATTERNS VISUALISATION
    viz_patterns = {
        'histogram': [r'histogram|distribution|distrib|frequence|freq'],
        'bar': [r'bar|barplot|bar chart|diagramme.*bar|bâtons?'],
        'line': [r'line|ligne|courbe|curve|tendance|trend|evolution|temporel'],
        'scatter': [r'scatter|nuage|point|correlation|relation|xy|croisement'],
        'pie': [r'pie|camembert|circulaire|proportion|part.*tout'],
        'box': [r'box|boite|boxplot|quartile|median.*dispersion'],
        'heatmap': [r'heatmap|correlation.*matrix|matrice.*corr']
    }
    
    for viz_name, patterns in viz_patterns.items():
        if any(re.search(p, query_lower) for p in patterns):
            intention = "visualisation"
            action = f"graphique {viz_name}"
            method = f"{viz_name}_plot()"
            params['chart_type'] = viz_name
            break
    
    # PATTERNS NETTOYAGE
    clean_patterns = {
        'missing': [r'manquant|missing|null|na|vide|absent'],
        'duplicate': [r'doublon|duplicate|repetition|redondant|identique'],
        'outlier': [r'outlier|anomalie|aberrant|extreme|atypique']
    }
    
    for clean_name, patterns in clean_patterns.items():
        if any(re.search(p, query_lower) for p in patterns):
            intention = "nettoyage"
            action = f"détection {clean_name}"
            method = f"detect_{clean_name}()"
            params['clean_type'] = clean_name
            break
    
    # PATTERNS GROUPBY ("par", "by")
    if re.search(r'\b(par|by|pour chaque|per|group.*by)\b', query_lower):
        # Trouver la colonne de groupement (catégorielle)
        group_cols = [c for c in detected_cols if col_types.get(c) in ['categorical', 'gender', 'location', 'category', 'binary']]
        value_cols = [c for c in detected_cols if col_types.get(c) in ['numeric', 'monetary', 'quantity', 'percentage']]
        
        if group_cols and value_cols:
            intention = "statistique" if intention == "exploration" else intention
            action = f"{action} groupé par {group_cols[0]}"
            method = f"groupby('{group_cols[0]}').{method}"
            params['groupby'] = group_cols[0]
            params['value_col'] = value_cols[0]
            # Réorganiser pour mettre la colonne de valeur en premier
            detected_cols = [value_cols[0], group_cols[0]]
        elif group_cols:
            params['groupby'] = group_cols[0]
    
    # PATTERNS PRÉDICTION
    pred_patterns = [r'predict|prediction|predire|forecast|prevoir|estimer|estimate|future|futur']
    if any(re.search(p, query_lower) for p in pred_patterns):
        intention = "prediction"
        action = "modèle prédictif"
        method = "regression/classification"
        params['target'] = detected_cols[0] if detected_cols else None
    
    # Construction de l'explication
    if detected_cols:
        col_desc = ", ".join([f"{c} ({col_types.get(c, 'inconnu')})" for c in detected_cols[:2]])
        explanation = f"Analyse '{action}' sur {col_desc}"
    else:
        explanation = f"Analyse générale : {action}"
    
    return {
        "intention": intention,
        "action": action,
        "colonnes_concernees": detected_cols if detected_cols else ["toutes"],
        "methode": method,
        "explication": explanation,
        "params": params,
        "types_detectes": {c: col_types.get(c, 'inconnu') for c in detected_cols[:2]}
    }

def generate_code(plan: Dict, df: pd.DataFrame, col_types: Dict[str, str]) -> str:
    """Génère le code Python selon le plan"""
    lines = []
    intent = plan['intention']
    cols = plan['colonnes_concernees']
    params = plan.get('params', {})
    
    if intent == 'statistique':
        stat = params.get('stat', 'mean')
        groupby = params.get('groupby')
        value_col = params.get('value_col', cols[0] if cols else df.columns[0])
        
        if groupby and value_col:
            if stat == 'mean':
                lines.append(f"result = df.groupby('{groupby}')['{value_col}'].mean().sort_values(ascending=False)")
            elif stat == 'sum':
                lines.append(f"result = df.groupby('{groupby}')['{value_col}'].sum().sort_values(ascending=False)")
            elif stat == 'count':
                lines.append(f"result = df.groupby('{groupby}').size()")
            else:
                lines.append(f"result = df.groupby('{groupby}')['{value_col}'].{stat}()")
        else:
            if cols[0] != 'toutes':
                lines.append(f"result = df['{cols[0]}'].{stat}()")
            else:
                lines.append(f"result = df.{stat}()")
    
    elif intent == 'visualisation':
        chart = params.get('chart_type', 'line')
        col = cols[0] if cols and cols[0] != 'toutes' else df.columns[0]
        col_type = col_types.get(col, 'unknown')
        
        lines.append("import plotly.express as px")
        lines.append("import plotly.graph_objects as go")
        
        if chart == 'histogram':
            lines.append(f"fig = px.histogram(df, x='{col}', title='Distribution de {col}', color_discrete_sequence=['#3366cc'])")
        elif chart == 'bar':
            if col_type in ['categorical', 'gender', 'location']:
                lines.append(f"value_counts = df['{col}'].value_counts().head(20)")
                lines.append(f"fig = px.bar(x=value_counts.index, y=value_counts.values, labels={{'x': '{col}', 'y': 'Count'}}, title='Répartition de {col}')")
            else:
                lines.append(f"fig = px.bar(df, x=df.index[:50], y='{col}', title='Valeurs de {col}')")
        elif chart == 'pie':
            lines.append(f"value_counts = df['{col}'].value_counts().head(10)")
            lines.append(f"fig = px.pie(names=value_counts.index, values=value_counts.values, title='Répartition de {col}')")
        elif chart == 'box':
            lines.append(f"fig = px.box(df, y='{col}', title='Boxplot de {col}')")
        elif chart == 'scatter' and len(cols) >= 2:
            lines.append(f"fig = px.scatter(df, x='{cols[0]}', y='{cols[1]}', title='{cols[0]} vs {cols[1]}', opacity=0.6)")
        else:
            lines.append(f"fig = px.line(df, y='{col}', title='Évolution de {col}')")
        
        lines.append("st.plotly_chart(fig, use_container_width=True)")
        lines.append("result = 'Graphique généré'")
    
    elif intent == 'nettoyage':
        clean_type = params.get('clean_type', 'missing')
        if clean_type == 'missing':
            lines.append("result = df.isnull().sum()")
            lines.append("result = result[result > 0]")
            lines.append("if len(result) == 0: result = 'Aucune valeur manquante'")
        elif clean_type == 'duplicate':
            lines.append("result = df.duplicated().sum()")
            lines.append("result = f'{result} doublons trouvés'")
        else:
            lines.append(f"result = df['{cols[0]}'].describe()")
    
    else:  # exploration
        lines.append("result = df.describe()")
    
    return "\n".join(lines)

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
