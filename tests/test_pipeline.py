import os
import pandas as pd

def test_cleaned_output_exists():
    assert os.path.exists("data/processed/accidents_cleaned.csv")

def test_required_columns():
    df = pd.read_csv("data/processed/accidents_cleaned.csv")
    for col in ["accident_id", "date", "severity"]:
        assert col in df.columns
