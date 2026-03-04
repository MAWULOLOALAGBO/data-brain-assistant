"""Tests pour le module loader"""

import pytest
import pandas as pd
from io import BytesIO
from core.loader import (
    detect_encoding, detect_separator, 
    load_csv, load_json, infer_types, get_column_info
)


def test_detect_separator():
    sample_comma = "a,b,c\n1,2,3"
    sample_semicolon = "a;b;c\n1;2;3"
    
    assert detect_separator(sample_comma) == ','
    assert detect_separator(sample_semicolon) == ';'


def test_load_csv_simple():
    content = b"name,age,city\nAlice,30,Paris\nBob,25,London"
    df = load_csv(content)
    
    assert len(df) == 2
    assert list(df.columns) == ['name', 'age', 'city']
    assert df['name'].iloc[0] == 'Alice'


def test_load_json_list():
    content = b'[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
    df = load_json(content)
    
    assert len(df) == 2
    assert 'name' in df.columns
    assert 'age' in df.columns


def test_infer_types_datetime():
    df = pd.DataFrame({
        'dates': ['2024-01-15', '2024-02-20', '2024-03-25'],
        'values': ['1', '2', '3']
    })
    
    df_typed = infer_types(df, aggressive=True)
    
    # dates devrait être datetime
    assert pd.api.types.is_datetime64_any_dtype(df_typed['dates'])
    # values devrait être numérique (avec aggressive=True)
    assert pd.api.types.is_numeric_dtype(df_typed['values'])


def test_get_column_info():
    df = pd.DataFrame({
        'A': [1, 2, 3, None],
        'B': ['x', 'y', 'z', 'x']
    })
    
    info = get_column_info(df)
    
    assert len(info) == 2
    assert info['null'].iloc[0] == 1  # Une valeur null dans A
    assert info['unique'].iloc[1] == 3  # 3 valeurs uniques dans B
