import pandas as pd

import psycopg2
from sqlalchemy import create_engine, text

from pathlib import Path

import pymatch


### Set pathnames
utils_dir = Path(__file__).parent.absolute()
data_dir = Path(__file__).parent.parent.parent.absolute() / "data"
sql_query_dir = utils_dir / "SQL queries"

### Setup postrges connection
pg_engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/mimic4"
)


def run_sql_from_txt(file_name, engine):
    """Takes a series of SQL statements from a .txt that do not return an output and runs them using the engine or connection specified.

    :param file_name: Title of SQL .txt file name stored in dir 'SQL queries'
    :type file_name: str
    :param engine: Established SQLalchemy engine to connect to relevant database
    :type engine: SQLalchemy Engine()
    """
    with open(sql_query_dir / file_name) as file:
        statements = file.read().replace("\n", " ").split(";")

    with engine.connect() as connection:
        for statement in statements:
            connection.execute(text(statement))


def load_sql_from_text(file_name, engine, **kwargs):
    """Run SQL queries stored at .txt files using specified SQL engine and return the relevant data.

    :param file_name: Title of SQL .txt file name stored in dir 'SQL queries'
    :type file_name: str
    :param engine: Established SQLalchemy engine to connect to relevant database
    :type engine: SQLalchemy Engine()
    :param col_dtypes: Dict of datatypes for returned dataframe columns
    :type col_dtypes: dict
    :return: DataFrame of return SQL queries
    :rtype: Pandas DataFrame
    """
    with open(sql_query_dir / file_name) as file:
        query = file.read().replace("\n", " ").replace("%", "%%")

    results = pd.read_sql_query(query, engine, **kwargs)
    return results


### Load Cases from pgAdmin
cases = load_sql_from_text("liver_cancer_patients.txt", engine=pg_engine)
cases.to_csv(data_dir / "input/cases.csv")
### Load controls from pgAdmin
controls = load_sql_from_text("non_liver_cancer_patients.txt", engine=pg_engine)
controls.to_csv(data_dir / "input/controls.csv")
### Load characteristics of MIMICIV cohort
run_sql_from_txt("characteristics.txt", pg_engine)
characteristics = pd.read_sql_query(
    "Select * From mimiciv_derived.characteristics", pg_engine
)
characteristics.to_csv(data_dir / "input/characteristics.csv")

### Match cohort
