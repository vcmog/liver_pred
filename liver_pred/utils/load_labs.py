import pandas as pd
import numpy as np

# import psycopg2
from sqlalchemy import create_engine, text, types

from pathlib import Path

from functions import run_sql_from_txt, load_sql_from_text

# from sklearn.linear_model import LogisticRegression

# import seaborn as sns
import matplotlib.pyplot as plt

import config

data_dir = config.data_dir

if config.perform_matching:
    cohort_ids = pd.read_csv(data_dir / "interim/matched_cohort_ids.csv", index_col=0)
else:
    cohort_ids = pd.read_csv(data_dir / "interim/cohort_ids.csv", index_col=0)

engine = create_engine(config.sql_connection_str)

# Load data
print("Loading data...")
lab_data = load_sql_from_text(
    "extract_labs.txt",
    engine=engine,
    dtype={
        "subject_id": "Int64",
        "index_admission": "Int64",
        "test_admission": "datetime64[ns]",
        "itemid": "Int64",
        "valuenum": "float64",
        "valueuom": "string",
        "flag": "string",
        "charttime": "datetime64[ns]",
        "outcome": "boolean",
        "index_date": "datetime64[ns]",
    },
)
print("Lab data loaded")
