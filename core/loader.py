"""
Module de chargement intelligent de fichiers.
Supporte : CSV, Excel, JSON, Parquet, TXT avec détection automatique.
"""

import pandas as pd
import json
import chardet
from io import StringIO, BytesIO
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings


def detect_encoding(file_content: bytes) -> str:
    """Détecte l'encodage d'un fichier binaire."""
    result = chardet.detect(file_content)
    return result.get('encoding', 'utf-8') or 'utf-8'


def detect_separator(sample: str) -> str:
    """Détecte le séparateur le plus probable dans un CSV."""
    separators = [',', ';', '\t', '|', ':']
    counts = {sep: sample.count(sep) for sep in separators}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ','


def load_csv(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Charge un CSV avec détection automatique d'encodage et séparateur."""
    # Détection encodage
    encoding = detect_encoding(file_content)
    
    # Lecture d'un échantillon pour détecter le séparateur
    try:
        sample = file_content[:4096].decode(encoding)
    except UnicodeDecodeError:
        encoding = 'latin-1'
        sample = file_content[:4096].decode(encoding, errors='ignore')
    
    separator = detect_separator(sample)
    
    # Lecture complète
    try:
        df = pd.read_csv(
            StringIO(file_content.decode(encoding)),
            sep=separator,
            engine='python',
            on_bad_lines='warn',
            **kwargs
        )
    except Exception as e:
        # Fallback : lecture avec pandas natif
        df = pd.read_csv(
            BytesIO(file_content),
            encoding=encoding,
            on_bad_lines='skip'
        )
    
    return df


def load_excel(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Charge un fichier Excel (.xlsx, .xls)."""
    try:
        # Essai avec openpyxl (plus récent)
        df = pd.read_excel(BytesIO(file_content), engine='openpyxl', **kwargs)
    except Exception:
        try:
            # Fallback xlrd pour .xls anciens
            df = pd.read_excel(BytesIO(file_content), engine='xlrd', **kwargs)
        except Exception as e:
            raise ValueError(f"Impossible de lire le fichier Excel: {e}")
    
    return df


def load_json(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Charge un JSON avec gestion des formats variés."""
    content = file_content.decode('utf-8', errors='ignore')
    
    # Essai 1 : JSON standard (objet ou liste)
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            # JSON avec une seule racine (ex: {"data": [...]})
            if len(data) == 1:
                key = list(data.keys())[0]
                if isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            # JSON normalisé
            return pd.json_normalize(data)
        elif isinstance(data, list):
            return pd.DataFrame(data)
    except json.JSONDecodeError:
        pass
    
    # Essai 2 : JSONL (une ligne par objet JSON)
    try:
        lines = [line for line in content.strip().split('\n') if line.strip()]
        data = [json.loads(line) for line in lines]
        return pd.DataFrame(data)
    except json.JSONDecodeError:
        pass
    
    # Essai 3 : JSON avec pandas direct
    try:
        return pd.read_json(StringIO(content))
    except Exception:
        pass
    
    raise ValueError("Format JSON non reconnu ou mal formé")


def load_parquet(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Charge un fichier Parquet."""
    return pd.read_parquet(BytesIO(file_content), **kwargs)


def load_text(file_content: bytes, **kwargs) -> pd.DataFrame:
    """Charge un fichier texte brut (une ligne = une entrée)."""
    try:
        # Essai CSV d'abord
        return load_csv(file_content, **kwargs)
    except Exception:
        # Fallback : texte brut
        content = file_content.decode('utf-8', errors='ignore')
        lines = content.strip().split('\n')
        return pd.DataFrame({'content': lines})


def load_file(file_obj, filename: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Charge n'importe quel fichier et retourne (DataFrame, metadata).
    
    Args:
        file_obj: Fichier uploadé (StreamlitUploadedFile) ou bytes
        filename: Nom du fichier (pour détection d'extension)
    
    Returns:
        Tuple (DataFrame, metadata_dict)
    """
    # Lecture du contenu
    if hasattr(file_obj, 'read'):
        file_content = file_obj.read()
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        filename = filename or getattr(file_obj, 'name', 'unknown')
    else:
        file_content = file_obj
        filename = filename or 'unknown'
    
    filename_lower = filename.lower()
    extension = Path(filename).suffix.lower()
    
    metadata = {
        'filename': filename,
        'extension': extension,
        'size_bytes': len(file_content),
        'detected_format': None
    }
    
    # Dispatch selon l'extension
    try:
        if extension in ['.csv', '.tsv', '.txt'] or filename_lower.endswith(('.csv', '.tsv')):
            df = load_csv(file_content)
            metadata['detected_format'] = 'CSV/TSV'
            
        elif extension in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
            df = load_excel(file_content)
            metadata['detected_format'] = 'Excel'
            
        elif extension in ['.json', '.jsonl']:
            df = load_json(file_content)
            metadata['detected_format'] = 'JSON'
            
        elif extension in ['.parquet', '.pq']:
            df = load_parquet(file_content)
            metadata['detected_format'] = 'Parquet'
            
        else:
            # Détection par contenu si extension inconnue
            try:
                df = load_csv(file_content)
                metadata['detected_format'] = 'CSV (auto-detect)'
            except Exception:
                try:
                    df = load_json(file_content)
                    metadata['detected_format'] = 'JSON (auto-detect)'
                except Exception:
                    df = load_text(file_content)
                    metadata['detected_format'] = 'Text (fallback)'
        
        metadata['rows'] = len(df)
        metadata['columns'] = len(df.columns)
        
        return df, metadata
        
    except Exception as e:
        raise ValueError(f"Erreur de chargement ({metadata.get('detected_format', 'unknown')}): {str(e)}")


def infer_types(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
    """
    Infère et convertit les types de données de manière intelligente.
    
    Args:
        df: DataFrame à analyser
        aggressive: Si True, tente des conversions plus risquées
    
    Returns:
        DataFrame avec types optimisés
    """
    df = df.copy()
    type_conversions = {}
    
    for col in df.columns:
        original_type = df[col].dtype
        new_type = None
        
        # Skip si déjà typé correctement
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        # 1. Détection datetime
        if df[col].dtype == 'object':
            # Échantillon non-null
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                # Heuristique : contient des chiffres et des séparateurs de date ?
                sample_str = sample.astype(str)
                date_patterns = sample_str.str.contains(r'\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}', regex=True)
                
                if date_patterns.mean() > 0.5:  # >50% ressemblent à des dates
                    try:
                        parsed = pd.to_datetime(df[col], errors='coerce')
                        if parsed.notna().sum() / len(df) > 0.8:  # >80% parsable
                            df[col] = parsed
                            new_type = 'datetime'
                            type_conversions[col] = f'object → datetime'
                            continue
                    except Exception:
                        pass
        
        # 2. Détection numérique
        if df[col].dtype == 'object' and aggressive:
            # Nettoyage : remplacer virgule par point, espaces
            cleaned = df[col].astype(str).str.replace(',', '.', regex=False)
            cleaned = cleaned.str.replace(r'\s+', '', regex=True)
            
            # Test conversion
            try:
                numeric = pd.to_numeric(cleaned, errors='coerce')
                if numeric.notna().sum() / len(df) > 0.8:  # >80% convertible
                    df[col] = numeric
                    new_type = 'numeric'
                    type_conversions[col] = f'object → numeric'
                    continue
            except Exception:
                pass
        
        # 3. Détection catégorielle (mémoire)
        if df[col].dtype == 'object':
            n_unique = df[col].nunique()
            n_total = len(df)
            
            # Ratio faible et pas trop de catégories uniques
            if n_unique / n_total < 0.5 and n_unique < 1000:
                df[col] = df[col].astype('category')
                new_type = 'category'
                type_conversions[col] = f'object → category'
        
        # 4. Optimisation des types numériques (mémoire)
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Stockage des conversions dans les attributs
    df.attrs['type_conversions'] = type_conversions
    
    return df


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un résumé détaillé des colonnes du DataFrame.
    """
    info = []
    
    for col in df.columns:
        col_data = {
            'colonne': col,
            'type': str(df[col].dtype),
            'non_null': df[col].count(),
            'null': df[col].isnull().sum(),
            'unique': df[col].nunique(),
            'memory_mb': df[col].memory_usage(deep=True) / 1024**2,
        }
        
        # Stats selon le type
        if pd.api.types.is_numeric_dtype(df[col]):
            col_data.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std(),
            })
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_data.update({
                'min': df[col].min(),
                'max': df[col].max(),
            })
        else:
            col_data.update({
                'min': None,
                'max': None,
                'mean': None,
                'std': None,
            })
        
        info.append(col_data)
    
    return pd.DataFrame(info)
