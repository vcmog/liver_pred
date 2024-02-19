import pandas as pd
import numpy as np

# import psycopg2
from sqlalchemy import create_engine, text

from pathlib import Path

from functions import run_sql_from_txt, load_sql_from_text

import config


import matplotlib.pyplot as plt

from PropensityScoreMatcher import PropensityScoreMatcher

# Set pathnames
utils_dir = config.utils_dir  # Path(__file__).parent.absolute()
project_dir = config.project_dir  # utils_dir.parent.parent.absolute()
data_dir = config.data_dir  # project_dir / "data"
sql_query_dir = config.sql_query_dir  # utils_dir / "SQL queries"
output_dir = config.output_dir  # project_dir / "outputs"

# Load data #

# Setup postrges connection
pg_engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/mimic4"
)


# Load Cases
print("Loading cases...")
if Path(data_dir / "input/cases.csv").exists():
    cases = pd.read_csv(data_dir / "input/cases.csv")
else:
    cases = load_sql_from_text("liver_cancer_patients.txt", engine=pg_engine)
    cases.to_csv(data_dir / "input/cases.csv")
# Load Controls
print("Loading controls...")
if Path(data_dir / "input/controls.csv").exists():
    controls = pd.read_csv(data_dir / "input/controls.csv")
else:
    controls = load_sql_from_text("non_liver_cancer_patients.txt", engine=pg_engine)
    controls.to_csv(data_dir / "input/controls.csv")

# Load characteristics of MIMICIV cohort
print("Loading characteristics...")
if Path(data_dir / "input/characteristics.csv").exists():
    characteristics = pd.read_csv(data_dir / "input/characteristics.csv")
else:
    run_sql_from_txt("characteristics.txt", pg_engine)
    characteristics = pd.read_sql_query(
        "Select * From mimiciv_derived.characteristics", pg_engine
    )
    characteristics.to_csv(data_dir / "input/characteristics.csv")


# Match cohort #

if config.perform_matching:
    print("Matching cohort...")
    case_chars = characteristics[
        characteristics["hadm_id"].isin(cases["hadm_id"])
    ].copy()
    control_chars = characteristics[
        characteristics["subject_id"].isin(controls["subject_id"])
    ].copy()
    # sample random admission for each patient so duplicate patients are included
    control_chars = (
        control_chars.groupby("subject_id")
        .apply(lambda x: x.sample(1))
        .reset_index(drop=True)
        .copy()
    )

    # option for number of matches
    no_matches = config.no_matches

    # Add outcome column to characteristics
    case_chars["outcome"] = 1
    control_chars["outcome"] = 0

    matcher = PropensityScoreMatcher(
        case_chars,
        control_chars,
        yvar="outcome",
        exclude=["subject_id", "hadm_id"],
    )

    print("Fitting models...")
    matcher.fit_score()
    print("Predicting scores...")
    matcher.predict_scores()
    print("Scores predicted")
    matcher.plot_scores(
        save_fig=True, save_path=output_dir / "cohort_matching/pre_match_scores.png"
    )
    plt.show()
    matcher.tune_threshold(
        save_fig=True, save_path=output_dir / "cohort_matching/threshold_plot.png"
    )
    plt.show()
    matcher.match(nmatches=5)
    print("Matching complete")
    matched_data = matcher.matched_data

    cohort_ids = matched_data[["subject_id", "hadm_id", "outcome"]]

    cohort_ids = pd.merge(
        cohort_ids, cases[["hadm_id", "index_date"]], on="hadm_id", how="left"
    )
    cohort_ids = pd.merge(
        cohort_ids, controls[["hadm_id", "index_date"]], on="hadm_id", how="left"
    )
    cohort_ids["index_date"] = cohort_ids["index_date_x"].combine_first(
        cohort_ids["index_date_y"]
    )
    cohort_ids = cohort_ids.drop(columns=["index_date_x", "index_date_y"])
    cohort_ids.to_csv(data_dir / "interim/matched_cohort_ids.csv")

    post_match = PropensityScoreMatcher(
        matched_data[matched_data["outcome"] == 1],
        matched_data[matched_data["outcome"] == 0],
        yvar="outcome",
        exclude=[
            "subject_id",
            "hadm_id",
            "scores",
            "match_id",
            "record_id",
        ],
    )
    post_match.fit_score()
    with open(output_dir / "cohort_matching/report.txt", "w+") as f:
        f.write(
            f"Prematching: \n ---------------\ncasen = {matcher.casen} \
                \ncontroln = {matcher.controln} \
                \naccuracy = {matcher.average_accuracy} \n\n\n \
        Postmatching: \n ---------------\ncasen = {post_match.casen} \
            \ncontroln = {post_match.controln} \
            \naccuracy = {post_match.average_accuracy}"
        )
else:
    cohort_ids = pd.concat([cases, controls])[
        ["subject_id", "hadm_id", "outcome"]
    ].sample(frac=1)

    cohort_ids = pd.merge(
        cohort_ids, cases[["hadm_id", "index_date"]], on="hadm_id", how="left"
    )
    cohort_ids = pd.merge(
        cohort_ids, controls[["hadm_id", "index_date"]], on="hadm_id", how="left"
    )
    cohort_ids["index_date"] = cohort_ids["index_date_x"].combine_first(
        cohort_ids["index_date_y"]
    )
    cohort_ids = cohort_ids.drop(columns=["index_date_x", "index_date_y"])
    cohort_ids.to_csv(data_dir / "interim/cohort_ids.csv")

print("Cohort IDs identified")
