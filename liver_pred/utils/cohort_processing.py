import pandas as pd
import numpy as np

# import psycopg2
from sqlalchemy import create_engine, text

from pathlib import Path


# from sklearn.linear_model import LogisticRegression

# import seaborn as sns
import matplotlib.pyplot as plt

from PropensityScoreMatcher import PropensityScoreMatcher

# Set pathnames
utils_dir = Path(__file__).parent.absolute()
project_dir = utils_dir.parent.parent.absolute()
data_dir = project_dir / "data"
sql_query_dir = utils_dir / "SQL queries"


## Load data ######

## Setup postrges connection
pg_engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/mimic4"
)


def run_sql_from_txt(file_name, engine):
    """Takes a series of SQL statements from a .txt that do not return an
     output and runs them using the engine or connection specified.

    :param file_name: Title of SQL .txt file name stored in dir 'SQL queries'
    :type file_name: str
    :param engine: Established SQLalchemy engine to connect to relevant
    database
    :type engine: SQLalchemy Engine()
    """
    with open(sql_query_dir / file_name) as file:
        statements = file.read().replace("\n", " ").split(";")

    with engine.connect() as connection:
        for statement in statements:
            connection.execute(text(statement))


def load_sql_from_text(file_name, engine, **kwargs):
    """Run SQL queries stored at .txt files using specified SQL engine and
     return the relevant data.

    :param file_name: Title of SQL .txt file name stored in dir 'SQL queries'
    :type file_name: str
    :param engine: Established SQLalchemy engine to connect to relevant
    database
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


### Load Cases
if Path(data_dir / "input/cases.csv").exists():
    cases = pd.read_csv(data_dir / "input/cases.csv")
else:
    cases = load_sql_from_text("liver_cancer_patients.txt", engine=pg_engine)
    cases.to_csv(data_dir / "input/cases.csv")
### Load Controls
if Path(data_dir / "input/controls.csv").exists():
    ccontrols = pd.read_csv(data_dir / "input/controls.csv")
else:
    controls = load_sql_from_text("non_liver_cancer_patients.txt", engine=pg_engine)
    controls.to_csv(data_dir / "input/controls.csv")

### Load characteristics of MIMICIV cohort
if Path(data_dir / "input/characteristics.csv").exists():
    characteristics = pd.read_csv(data_dir / "input/characteristics.csv")
else:
    run_sql_from_txt("characteristics.txt", pg_engine)
    characteristics = pd.read_sql_query(
        "Select * From mimiciv_derived.characteristics", pg_engine
    )
    characteristics.to_csv(data_dir / "input/characteristics.csv")


###### Match cohort ######


## Select a random admission for each subject_id - otherwise can't work out with pymatch how to ensure the same patient isn't picked loads of times (could do this by scratch)
## this is a bad idea - can only do this for controls
characteristics = (
    characteristics.groupby("subject_id")
    .apply(lambda x: x.sample(1))
    .reset_index(drop=True)
)

## Add outcome column to characteristics
case_ids = cases["subject_id"]
characteristics["outcome"] = characteristics["subject_id"].isin(case_ids).astype(int)


## Match cases and cohorts to their characteristics
case_characteristics = characteristics[characteristics["outcome"] == 1]
control_characteristics = characteristics[characteristics["outcome"] == 0]

combined_df = pd.concat([case_characteristics, control_characteristics])
## Fit initial model

controlled_characteristics = combined_df[
    ["admission_type", "gender", "race", "age_at_admission", "rank"]
]

# one-hot-encode categorical variables - admission type, gender, race
controlled_characteristics_dummies = pd.get_dummies(
    controlled_characteristics, drop_first=True
)

outcome = combined_df["outcome"]


matcher = PropensityScoreMatcher(
    combined_df[combined_df["outcome"] == 1],
    combined_df[combined_df["outcome"] == 0],
    yvar="outcome",
    exclude=["subject_id", "hadm_id"],
)


matcher.fit_score()
print("Scores fit")
print(matcher.data)
print(matcher.X)
matcher.predict_scores()
print("Scores predicted")
matcher.plot_scores()
print("done")
plt.show()
