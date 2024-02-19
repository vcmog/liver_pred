import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from pathlib import Path


def progress(i, n, prestr=""):
    sys.stdout.write("\r{}: {}\{}".format(prestr, i, n))


def run_sql_from_txt(
    file_name, engine, sql_query_dir=Path(__file__).parent.absolute() / "SQL queries"
):
    """Takes a series of SQL statements from a .txt that do not return an
     output and runs them using the engine or connection specified.

    :param file_name: Title of SQL .txt file name stored in dir 'SQL queries'
    :type file_name: str
    :param engine: Established SQLalchemy engine to connect to relevant
    database
    :type engine: SQLalchemy Engine()
    :sql_query_dir: Name of SQL query directory
    :type sql_query_dir: str
    """
    with open(sql_query_dir / file_name) as file:
        statements = file.read().replace("\n", " ").split(";")

    with engine.connect() as connection:
        for statement in statements:
            connection.execute(text(statement))


def load_sql_from_text(
    file_name,
    engine,
    sql_query_dir=Path(__file__).parent.absolute() / "SQL queries",
    **kwargs
):
    """Run SQL queries stored at .txt files using specified SQL engine and
     return the relevant data.

    :param file_name: Title of SQL .txt file name stored in dir 'SQL queries'
    :type file_name: str
    :param engine: Established SQLalchemy engine to connect to relevant
    database
    :type engine: SQLalchemy Engine()
    :sql_query_dir: Name of SQL query directory
    :type sql_query_dir: str
    :param col_dtypes: Dict of datatypes for returned dataframe columns
    :type col_dtypes: dict
    :return: DataFrame of return SQL queries
    :rtype: Pandas DataFrame
    """
    with open(sql_query_dir / file_name) as file:
        query = file.read().replace("\n", " ").replace("%", "%%")

    results = pd.read_sql_query(query, engine, **kwargs)

    return results
