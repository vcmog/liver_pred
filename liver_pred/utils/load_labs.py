import pandas as pd

# import psycopg2
from sqlalchemy import create_engine

from pathlib import Path

from functions import load_sql_from_text

import config

data_dir = config.data_dir

if config.perform_matching:
    cohort_ids = pd.read_csv(data_dir / "interim/matched_cohort_ids.csv", index_col=0)
else:
    cohort_ids = pd.read_csv(data_dir / "interim/cohort_ids.csv", index_col=0)

engine = create_engine(config.sql_connection_str)
conn = engine.connect()  # .execution_options(stream_results=True)

# Load data
print("Loading data...")
lab_data = load_sql_from_text(
    "extract_labs.txt",
    engine=conn,
    dtype={
        "subject_id": "Int64",
        "index_admission": "Int64",
        "test_admission": "Int64",
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
lab_data.to_csv(data_dir / "interim/lab_events")
