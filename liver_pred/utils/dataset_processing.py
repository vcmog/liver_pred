import pandas as pd

import psycopg2
from sqlalchemy import create_engine, text

from pathlib import Path


### Set pathnames
utils_dir = Path(__file__).parent.absolute()
data_dir = Path(__file__).parent.parent.parent.absolute() / "data"
sql_query_dir = utils_dir / "SQL queries"

### Setup postrges connection
engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/mimic4")


def run_sql_from_txt(file_name, engine):
    """Takes a series of SQL statements from a .txt that do not return an output and runs them using the engine or connection specified.

    :param file_name: Title of SQL .txt file name stored in dir 'SQL queries'
    :type file_name: str
    :param engine: Established SQLalchemy engine to connect to relevant database
    :type engine: SQLaclhemy Engine()
    """
    with open(sql_query_dir / file_name) as file:
        statements = file.read().replace("\n", " ").split(";")

    with engine.connect() as connection:
        for statement in statements:
            result = connection.execute(text(statement))


def load_sql_from_text(file_name, engine, col_dtypes=None):
    with open(sql_query_dir / file_name) as file:
        query = file.read().replace("\n", " ")

    results = pd.read_sql_query(query, engine, dtype=col_dtypes)
    return results


### Load Cases from pgAdmin


### Load controls from pgAdmin


### Load characteristics of MIMICIV cohort


### Match cohort
