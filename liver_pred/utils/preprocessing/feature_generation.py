import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import linregress
import utils.preprocessing.missing_data as md
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

    # for var_name in variable_names:
    #    if (
    #        var_name not in df.columns
    #        and var_name != "outcome"
    #        and var_name != "subject_id"
    #    ):
    #        # If variable name doesn't exist as a column, add it with NaN values
    #        df[var_name] = (
    #            np.nan
    #        )  # Or you can use df[var_name] = None for object columns
    missing_columns = {
        var_name: np.nan
        for var_name in variable_names
        if var_name not in df.columns and var_name not in {"outcome", "subject_id"}
    }
    df = df.assign(**missing_columns)


def lab_within_n_days(lab_df, n_days_pre, n_days_post, use_pseudo_index=False):
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
    if use_pseudo_index:
        index_col = "pseudo_index"
    else:
        index_col = "index_date"

    labs_within_n_days = lab_df[
        (lab_df["charttime"] < (lab_df[index_col] + d2))
        & (lab_df["charttime"] > (lab_df[index_col] - d1))
    ]
    return labs_within_n_days


def current_bloods_df(lab_df, lead_time=0, n_days_pre=7, n_days_post=1):
    """
    Generate a dataframe of current blood test results for each subject.

    Args:
        lab_df (pandas.DataFrame): The input dataframe containing blood test results.
        lead_time (int, optional): The number of days to subtract from the index date
        for early detection. Defaults to 0.
        n_days_pre (int, optional): The number of days before the index date to include
        in current_df. Defaults to 7.
        n_days_post (int, optional): The number of days after the index date to include
        in the current_df. Defaults to 1.

    Returns:
        pandas.DataFrame: The generated dataframe of current blood test results for
        each subject.
    """
    unique_ids = pd.DataFrame({"subject_id": lab_df["subject_id"].unique()})
    unique_ids["outcome"] = (
        lab_df.groupby(["subject_id"], observed=False)["outcome"].agg("max").values
    )

    if lead_time:
        lab_df["pseudo_index"] = lab_df["index_date"] - timedelta(days=lead_time)
        use_pseudo = True
    else:
        use_pseudo = False

    current = lab_within_n_days(
        lab_df,
        n_days_pre=n_days_pre,
        n_days_post=n_days_post,
        use_pseudo_index=use_pseudo,
    )

    # Find the mean value for each lab test
    current = (
        current[["subject_id", "label", "valuenum"]]
        .groupby(["subject_id", "label"], observed=False)[["valuenum"]]
        .mean()
    )

    # separate outcomes into a different column so can pivot the lab_df so that each variable is a column
    # outcomes = current["outcome"].groupby("subject_id").agg("max")
    current = current.pivot_table(
        index="subject_id",
        columns="label",
        values="valuenum",
        observed=False,
        dropna=False,
    )
    # TO DO: add line to add back other subject_ids who have no values - or not?

    # check_and_add_columns(current, find_variables(lab_df))

    # Add rows for unique_ids not already in the index of current
    # missing_ids = list(set(unique_ids["subject_id"]) - set(current.index))
    # missing_data = pd.DataFrame(index=missing_ids, columns=current.columns)
    # current = pd.concat([current, missing_data])

    current["outcome"] = unique_ids.set_index("subject_id").loc[current.index][
        "outcome"
    ]
    return current.reset_index(names="subject_id")


def historical_labs(lab_df, lead_time=0, n_days=0):
    """
    Filter the lab_df DataFrame to include only historical lab records.

    Parameters:
    lab_df (DataFrame): The DataFrame containing lab records.
    lead_time (int, optional): The number of days to subtract from the index_date. Defaults to 0.
    n_days (int, optional): Number of days between the most recent historical measurement and the index. Defaults to 0.
    use_pseudo_index (bool, optional): Whether to use a pseudo index for filtering. Defaults to False.

    Returns:
    DataFrame: The filtered DataFrame containing historical lab records.
    """

    if lead_time:
        lab_df["pseudo_index"] = lab_df["index_date"] - timedelta(days=lead_time)
        index_col = "pseudo_index"
    else:
        index_col = "index_date"

    d = timedelta(days=n_days)
    historical_lab_df = lab_df[(lab_df["charttime"] < lab_df[index_col] - d)]
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
    if "pseudo_index" in lab_df.columns:
        index_col = "pseudo_index"
    else:
        index_col = "index_date"

    custom_agg = {
        index_col: "first",  # Keep the first value (assuming it's the same for each group)
        "valuenum": "mean",  # Aggregate 'valuenum' using the mean
    }
    grouped = (
        lab_df[["subject_id", "label", "valuenum", "charttime", index_col]]
        .groupby(["subject_id", "label", pd.Grouper(key="charttime", freq="W")])
        .agg(custom_agg)
    )

    third_level_data = grouped.index.get_level_values(2)
    grouped["differences"] = grouped[index_col] - third_level_data
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
        group_data = group_data[group_data["valuenum"] == group_data["valuenum"]]
        X = group_data["differences"].values
        y = group_data["valuenum"].values

        if len(X) > 1:
            model_results = linregress(X, y)

            month_6 = model_results.slope * proximal_window + model_results.intercept
            month_12 = model_results.slope * distal_window + model_results.intercept

            current_value = current_df[current_df["subject_id"] == subject_id][
                label
            ].values[0]

            distal_trend = month_12 - month_6
            proximal_trend = month_6 - current_value

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
    lead_time=0,
    current_window_preindex=3,
    current_window_postindex=1,
    historical_window=0,
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
        processed_labs,
        lead_time=lead_time,
        n_days_pre=current_window_preindex,
        n_days_post=current_window_postindex,
    )
    outcome = current_labs["outcome"]

    # Fill missing values in the current_labs dataframe
    current_labs = md.fill_nas_normal(current_labs.drop("outcome", axis=1))

    current_labs["outcome"] = outcome

    historical_lab_df = historical_labs(
        processed_labs, lead_time=lead_time, n_days=historical_window
    )

    # get rid of this behaviour
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


def create_array_for_CNN(processed_labs, lead_time=0, max_history=None):
    """
    Create a 3D array for Convolutional Neural Network (CNN) input.

    Args:
        processed_labs (DataFrame): Processed laboratory measurements.
        lead_time (int, optional): Lead time in days. Defaults to 0.
        max_history (int, optional): Maximum history to consider. Defaults to None.

    Returns:
        array_3d (ndarray): 3D array with dimensions (subject_id, blood_test_label, difference).
        outcome (Series): Series containing the outcome for each subject.

    """
    binned_df = bin_measurements(processed_labs)
    binned_df["weekly_differences"] = round(binned_df["differences"] // 7, 0)
    binned_df = binned_df[binned_df["weekly_differences"] > lead_time / 7]
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

    all_differences = range(0, int(max(binned_df["weekly_differences"])) + 1)

    multiindex = pd.MultiIndex.from_product(
        [pivot_df.index.levels[0], all_differences], names=["subject_id", "differences"]
    )

    pivot_df = pivot_df.reindex(multiindex)

    array_3d = pivot_df.values.reshape(
        (-1, len(pivot_df.columns), len(pivot_df.index.levels[1]))
    )
    array_3d = np.nan_to_num(array_3d)

    return array_3d, outcome.values.flatten()


def create_array_for_RNN(
    processed_labs, lead_time=0, max_history=None, pad=True, zero_fill_nan=True
):
    """
    Create an array for Recurrent Neural Network (RNN) input.

    Args:
        processed_labs (DataFrame): The processed labs data.
        lead_time (int, optional): The minimum time difference in days between index_date and charttime. Defaults to 0.
        max_history (int, optional): The maximum time difference in days between index_date and charttime. Defaults to None.
        pad (bool, optional): Whether to pad the input sequences. Defaults to True.

    Returns:
        array: The array for RNN input.

    Raises:
        None

    """

    processed_labs["differences"] = (
        processed_labs["index_date"] - processed_labs["charttime"]
    )

    processed_labs["differences"] = (
        processed_labs["differences"] / np.timedelta64(1, "D")
    ).astype(int)

    processed_labs = processed_labs.loc[processed_labs["differences"] > lead_time]
    if max_history:
        processed_labs = processed_labs.loc[processed_labs["differences"] < max_history]

    processed_labs.sort_values(["subject_id", "charttime"], inplace=True)
    processed_labs["time_diff"] = processed_labs.groupby("subject_id")[
        "charttime"
    ].diff().dt.total_seconds().fillna(0) / np.timedelta64(1, "D").astype(int)
    time_diff = processed_labs[["subject_id", "charttime", "time_diff"]]
    time_diff = time_diff.groupby(["subject_id", "charttime"]).mean()
    pivoted_df = processed_labs.pivot_table(
        index=["subject_id", "charttime"], columns="label", values="valuenum"
    )

    pivoted_df["time_diff"] = time_diff["time_diff"]

    subject_data = pivoted_df.groupby(level=0).apply(lambda x: list(x.to_numpy()))
    outcomes = (
        processed_labs[["subject_id", "outcome"]]
        .groupby("subject_id")
        .max()
        .loc[subject_data.index]
    )
    if zero_fill_nan:
        subject_data = subject_data.apply(lambda x: np.nan_to_num(x))
    if pad:
        max_len = max([len(x) for x in subject_data])
        padded_inputs = subject_data.apply(
            lambda x: np.pad(
                x, ((max_len - len(x), 0), (0, 0)), mode="constant", constant_values=0
            )
        )
        padded_inputs = np.dstack(padded_inputs.values).transpose((2, 0, 1))
        return np.array(padded_inputs), np.array(outcomes).flatten()
    else:
        return subject_data
