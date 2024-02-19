import pandas as pd
import numpy as np

# import psycopg2
from sqlalchemy import create_engine, text

from pathlib import Path

from functions import run_sql_from_txt, load_sql_from_text

# from sklearn.linear_model import LogisticRegression

# import seaborn as sns
import matplotlib.pyplot as plt

import config

data_dir = config.data_dir

cohort_ids = pd.read_csv(data_dir / "interim/matched_cohort_ids.csv")
print(cohort_ids.head())
