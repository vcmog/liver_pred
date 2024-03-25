import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import linregress
import utils.missing_data as md
import utils.config as config


def find_variables(lab_df):
    lab_variables = lab_df["label"].unique()
    return lab_variables


def check_and_add_columns(df, variable_names):
    """
    Check if the given variable names exist as columns in the DataFrame.
    If a variable name doesn't exist, add it as a column with NaN values.

    Args:
        df (pandas.DataFrame): The DataFrame to check and modify.
        variable_names (list): A list of variable names to check and add.

    Returns:
        None
    """

    for var_name in variable_names:
        if var_name not in df.columns:
            # If variable name doesn't exist as a column, add it with NaN values
            df[var_name] = (
                np.nan
            )  # Or you can use df[var_name] = None for object columns


def lab_within_n_days(lab_df, n_days_pre, n_days_post):
    """
    Returns a subset of lab_df containing lab measurements within a specified
    number of days from the index date.

    Parameters:
    lab_df (DataFrame): The dataframe containing lab measurements.
    n_days (int): The number of days within which the lab measurements
    should be considered.

    Returns:
    DataFrame: A subset of lab_df containing lab measurements within n_days
    from the index date.
    """
    d1 = timedelta(days=n_days_pre)
    d2 = timedelta(days=n_days_post)
    labs_within_n_days = lab_df[
        (lab_df["charttime"] < lab_df["index_date"] + d2)
        & (lab_df["charttime"] > lab_df["index_date"] - d1)
    ]
    return labs_within_n_days


def current_bloods_df(lab_df, n_days_pre=7, n_days_post=1):
    """
    Generate a dataframe of current blood test results for each subject.

    Args:
        lab_df (pandas.DataFrame): The input dataframe containing blood test results.
        n_days (int, optional): The number of days to consider for current blood test
        results. Defaults to 14.

    Returns:
        pandas.DataFrame: A dataframe with the mean value of each lab test for each
                          subject, with outcomes separated into a different column
                          and the lab tests pivoted so that each variable is a column.
    """
    current = lab_within_n_days(lab_df, n_days_pre=n_days_pre, n_days_post=n_days_post)
    # Find the mean value for each lab test
    current = current.groupby(["subject_id", "label"])[["valuenum", "outcome"]].agg(
        "mean"
    )
    # separate outcomes into a different column so can pivot the lab_df so that each variable is a column
    outcomes = current["outcome"].groupby("subject_id").agg("max")
    current = current.pivot_table(
        index="subject_id", columns="label", values="valuenum"
    )
    check_and_add_columns(current, find_variables(lab_df))
    current["outcome"] = outcomes
    return current.reset_index()


def historical_labs(lab_df, n_days=14):
    """
    Filter the lab_df dataframe to include only the historical lab records
    that are older than n_days.

    Parameters:
    lab_df (pandas.DataFrame): The dataframe containing lab records.
    n_days (int): The number of days to consider for historical lab records.
    Default is 14.

    Returns:
    pandas.DataFrame: The filtered dataframe containing historical lab records.
    """
    d = timedelta(days=n_days)
    historical_lab_df = lab_df[(lab_df["charttime"] < lab_df["index_date"] - d)]
    return historical_lab_df


def write_report(current_labs, historical_labs, dir, extra_strings=None):
    current_len = len(current_labs)
    current_control_len = len(current_labs[current_labs["outcome"] == 0])
    current_case_len = len(current_labs[current_labs["outcome"] == 1])
    with open(dir / "colonflag/historical_labs_report.txt", "w") as f:
        f.write(
            "Number of patients with measurements in current_labs:{} (total) {} (control) {} (case)\n".format(
                current_len,
                current_control_len,
                current_case_len,
            )
        )
        f.write(
            "Number of patients with measurements in historical_labs: {} (total) {} (control) {} (case)\n".format(
                historical_labs["subject_id"].nunique(),
                historical_labs[historical_labs["outcome"] == 0][
                    "subject_id"
                ].nunique(),
                historical_labs[historical_labs["outcome"] == 1][
                    "subject_id"
                ].nunique(),
            )
        )
        f.write(
            "Percentages of patients with historical_labs measurements: {:.2f}% (total) {:.2f}% (control) {:.2f}% (case)\n".format(
                (historical_labs["subject_id"].nunique() / current_len) * 100,
                (
                    historical_labs[historical_labs["outcome"] == 0][
                        "subject_id"
                    ].nunique()
                    / current_control_len
                )
                * 100,
                (
                    historical_labs[historical_labs["outcome"] == 1][
                        "subject_id"
                    ].nunique()
                    / current_case_len
                )
                * 100,
            )
        )
        if extra_strings:
            for string in extra_strings:
                f.write(string + "\n")


def bin_measurements(lab_df):
    """
    Divide sequence of a single measurement into bins.

    Args:
        lab_df (DataFrame): Data frame with an individual row for each recorded laboratory test. Columns must include:
            - 'subject_id': Unique identifier for each patient.
            - 'label': Laboratory test name or id, e.g. 'creatinine'.
            - 'valuenum': Laboratory test results, numeric value.
            - 'charttime': Recorded time of test measurement.
            - 'index_date': Index date of patient, e.g. recorded/chosen time of outcome.

    Returns:
        DataFrame: Grouped data frame with the following columns:
            - 'subject_id': Unique identifier for each patient.
            - 'label': Laboratory test name or id.
            - 'charttime': Recorded time of test measurement (grouped by week).
            - 'index_date': Index date of patient.
            - 'valuenum': Mean value of 'valuenum' for each group.
            - 'differences': Difference in days between 'index_date' and 'charttime'.

    """
    custom_agg = {
        "index_date": "first",  # Keep the first value (assuming it's the same for each group)
        "valuenum": "mean",  # Aggregate 'valuenum' using the mean
    }
    grouped = (
        lab_df[["subject_id", "label", "valuenum", "charttime", "index_date"]]
        .groupby(["subject_id", "label", pd.Grouper(key="charttime", freq="W")])
        .agg(custom_agg)
    )

    third_level_data = grouped.index.get_level_values(2)
    grouped["differences"] = grouped["index_date"] - third_level_data
    grouped["differences"] = (grouped["differences"] / np.timedelta64(1, "D")).astype(
        int
    )
    return grouped


def generate_trend_features(
    binned_df, current_df, proximal_window=180, distal_window=365
):
    """
    Generate proximal and distal trends for each subject and variable.

    Args:
        binned_df (pandas.DataFrame): The binned dataframe containing the data.
        current_df (pandas.DataFrame): The current dataframe containing the data.
        proximal_window (int, optional): The window size for the proximal trend calculation in days. Defaults to 180.
        distal_window (int, optional): The window size for the distal trend calculation in days. Defaults to 365.

    Returns:
        pandas.DataFrame: A dataframe containing the subject IDs, variables, distal trends, and proximal trends.
    """

    groups = binned_df.groupby(["subject_id", "label"])

    subject_ids = []
    variables = []
    distals = []
    proximals = []

    for (subject_id, label), group_data in groups:

        X = group_data["differences"].values
        y = group_data["valuenum"].values

        if len(X) > 1:
            model_results = linregress(X, y)

            month_6 = model_results.slope * proximal_window + model_results.intercept
            month_12 = model_results.slope * distal_window + model_results.intercept

            distal_trend = month_12 - month_6
            proximal_trend = month_6 - (
                current_df[current_df["subject_id"] == subject_id][label]
            )

        else:
            distal_trend = proximal_trend = 0
        subject_ids += [subject_id]
        variables += [label]
        distals += [distal_trend]
        proximals += [proximal_trend]

    trend = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "variable": variables,
            "distal_trend": distals,
            "proximal_trend": proximals,
        }
    )
    trends_pivoted = trend.pivot(
        index="subject_id",
        columns="variable",
        values=["distal_trend", "proximal_trend"],
    )

    trends_pivoted.columns = [f"{col[0]}_{col[1]}" for col in trends_pivoted.columns]
    for col in trends_pivoted.columns:
        trends_pivoted[col] = trends_pivoted[col].apply(
            lambda x: x.squeeze() if isinstance(x, pd.Series) else x
        )
    return trends_pivoted.reset_index()


def generate_features(
    processed_labs,
    cohort_ids,
    current_window_preindex=30,
    current_window_postindex=3,
    historical_window=30,
    proximal=config.proximal_timepoint,
    distal=config.distal_timepoint,
):
    """
    Generate features for liver cancer prediction.

    Args:
        processed_labs (DataFrame): Processed laboratory data.
        cohort_ids (DataFrame): Cohort IDs data.
        current_window_preindex (int, optional): Number of days before the current timepoint to include in the current labs data. Defaults to 30.
        current_window_postindex (int, optional): Number of days after the current timepoint to include in the current labs data. Defaults to 3.
        historical_window (int, optional): Number of days to include in the historical labs data. Defaults to 30.

    Returns:
        Tuple: A tuple containing the current labs data, trend features, and the final feature dataframe.
    """

    current_labs = current_bloods_df(
        processed_labs, current_window_preindex, current_window_postindex
    )

    outcome = current_labs["outcome"]

    # Fill missing values in the current_labs dataframe
    current_labs = md.fill_nas_normal(current_labs.drop("outcome", axis=1))

    current_labs["outcome"] = outcome

    historical_lab_df = historical_labs(processed_labs, historical_window)

    current_labs, historical_lab_df, removed = md.remove_if_missing_from_other(
        current_labs, historical_lab_df
    )
    print(f"Removed {len(removed)} subject_ids from current_labs and historical_labs")
    print(
        f"Removed {sum(cohort_ids[cohort_ids['subject_id'].isin(removed)]['outcome'])} positive outcomes from current_labs"
    )

    # historical_labs.to_csv(config.utils_dir / "historical_labs.csv")

    write_report(current_labs, historical_lab_df, config.output_dir)
    # Remove subject_ids from current_labs and historical_labs that are not present in both

    binned_labs = bin_measurements(historical_lab_df)

    # current_labs.to_csv(config.utils_dir / "current_labs.csv")
    # binned_labs.to_csv(config.utils_dir / "binned_labs.csv")
    trend_features = generate_trend_features(
        binned_labs, current_labs, proximal, distal
    )
    feature_df = pd.merge(current_labs, trend_features, on="subject_id")
    # move outcome to end
    column_to_move = feature_df.pop("outcome")
    feature_df.insert(0, "outcome", column_to_move)
    return current_labs, trend_features, feature_df


def create_array_for_CNN(processed_labs, current_window_postindex, max_history=None):
    """
    Create a 3D array for Convolutional Neural Network (CNN) input.

    Args:
        processed_labs (DataFrame): Processed laboratory measurements.
        current_window_postindex (int): Current window post index.
        max_history (int, optional): Maximum history to consider. Defaults to None.

    Returns:
        array_3d (ndarray): 3D array with dimensions (subject_id, blood_test_label, difference).

    """
    binned_df = bin_measurements(processed_labs)
    # outcome = processed_labs[["subject_id", "outcome"]].drop_duplicates()
    binned_df["weekly_differences"] = round(binned_df["differences"] // 7, 0)
    binned_df = binned_df[binned_df["weekly_differences"] > -current_window_postindex]
    if max_history:
        binned_df = binned_df[binned_df["weekly_differences"] < max_history]

    pivot_df = (
        binned_df.reset_index()
        .fillna(0)
        .pivot_table(
            index=["subject_id", "weekly_differences"],
            columns="label",
            values="valuenum",
        )
    )
    subject_ids = pivot_df.index.get_level_values(0).drop_duplicates()
    outcome = (
        processed_labs[["subject_id", "outcome"]]
        .drop_duplicates()
        .set_index("subject_id")
        .loc[subject_ids]["outcome"]
    )
    pivot_df = pivot_df.fillna(0)

    # Get all possible differences and blood test labels
    all_differences = range(0, int(max(binned_df["weekly_differences"])) + 1)

    # Create a MultiIndex with all possible combinations of difference and subject_id
    multiindex = pd.MultiIndex.from_product(
        [pivot_df.index.levels[0], all_differences], names=["subject_id", "differences"]
    )

    # Reindex the pivot table to ensure all combinations are present, filling missing values with NaN
    pivot_df = pivot_df.reindex(multiindex)

    # 3d array with dimensions (subject_id, blood_test_label, difference)
    array_3d = pivot_df.values.reshape(
        (-1, len(pivot_df.columns), len(pivot_df.index.levels[1]))
    )
    array_3d = np.nan_to_num(array_3d)

    return array_3d, outcome
