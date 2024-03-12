import pandas as pd
import numpy as np
import utils.config as config

if not config.utils_dir / "ref_ranges.csv":
    print("No reference range file found, upload using SQL query")
ref_ranges = pd.read_csv(config.utils_dir / "ref_ranges.csv")


def process_ref_ranges(ref_ranges):
    if "centre" in ref_ranges.columns:
        return ref_ranges
    ref_ranges["label"] = ref_ranges["fluid"] + " " + ref_ranges["label"]
    ref_ranges = (
        ref_ranges.groupby("label")
        .agg({"ref_range_lower": "min", "ref_range_upper": "max"})
        .reset_index()
    )
    ref_ranges["centre"] = round(
        (ref_ranges["ref_range_upper"] + ref_ranges["ref_range_lower"]) / 2, 2
    )
    ref_ranges.to_csv(config.utils_dir / "ref_ranges.csv")
    return ref_ranges


ref_ranges = process_ref_ranges(ref_ranges)


def compare_subject_ids(df1, df2):
    """
    Compare the subject IDs in two DataFrames.

    Parameters:
    df1 (pandas.DataFrame): The first DataFrame to compare.
    df2 (pandas.DataFrame): The second DataFrame to compare.

    Returns:
    list: The subject IDs that are present in both DataFrames.
    """
    return (
        list(set(df1["subject_id"]).intersection(set(df2["subject_id"]))),
        list(set(df1["subject_id"]).union(set(df2["subject_id"]))),
    )


def remove_if_missing_from_other(df1, df2):
    """
    Removes rows from df1 and df2 where the subject_id is missing in either dataframe.

    Args:
        df1 (pandas.DataFrame): The first dataframe.
        df2 (pandas.DataFrame): The second dataframe.

    Returns:
        tuple: A tuple containing the filtered df1, filtered df2, and the removed subject_ids.
    """
    common_subject_ids, all_subject_ids = compare_subject_ids(df1, df2)
    removed_subject_ids = [i for i in all_subject_ids if i not in common_subject_ids]
    return (
        df1[df1["subject_id"].isin(common_subject_ids)],
        df2[df2["subject_id"].isin(common_subject_ids)],
        removed_subject_ids,
    )


def fill_na_zero(df):
    """
    Fills missing values in the given DataFrame with zeros.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be filled.

    Returns:
    pandas.DataFrame: The DataFrame with missing values filled with zeros.
    """
    common_labs_missing_refs = []
    for col in df.columns:
        if "trend" in col:
            df[col].fillna(0, inplace=True)
        elif (
            ref_ranges[ref_ranges["label"] == col]["ref_range_upper"].isna()
            | ref_ranges[ref_ranges["label"] == col]["ref_range_lower"].isna()
        ):
            print("Ref range missing:", col)
            common_labs_missing_refs += [col]
        else:
            df[col].fillna(0, inplace=True)
    return df


def fill_na_midrange(df):
    """
    Fills missing values in the given DataFrame using the midpoint of the normal range.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be filled.

    Returns:
    pandas.DataFrame: The DataFrame with missing values filled using the midpoint of the normal range.
    """
    common_labs_missing_refs = []
    for col in df.columns:
        if "trend" in col:
            df[col].fillna(0, inplace=True)
        elif (
            ref_ranges[col]["ref_range_upper"].isna()
            | ref_ranges[col]["ref_range_lower"].isna()
        ):
            print("Ref range missing:", col)
            common_labs_missing_refs += [col]
        else:
            df[col].fillna(
                ref_ranges[ref_ranges["label"] == col]["centre"], inplace=True
            )
    return df


def fill_nas_mean(df):
    """
    Fill missing values in the given DataFrame using the mean value of each column.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data to be filled.

    Returns:
    pandas.DataFrame: The DataFrame with missing values filled using the mean value of each column.
    """
    common_labs_missing_refs = []
    for col in df.columns:
        if "trend" in col:
            df[col].fillna(0, inplace=True)
        elif (
            ref_ranges[ref_ranges["label"] == col]["ref_range_upper"].isna()
            | ref_ranges[ref_ranges["label"] == col]["ref_range_lower"].isna()
        ):
            print("Ref range missing:", col)
            common_labs_missing_refs += [col]
        else:
            df[col].fillna(df["col"].mean(skipna=True), inplace=True)
    return df


def fill_nas_normal(df):
    """
    Fill missing values in a DataFrame using normal distribution sampling between the normal range,
    with the lower and upper reference range point forming the upper and lower bound of the 95% conf.
    interval.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing missing values to be filled.

    Returns:
    - df (pandas.DataFrame): The DataFrame with missing values filled using normal distribution sampling.
    """
    common_labs_missing_refs = []
    for col in df.columns:
        if "trend" in col:
            df[col].fillna(0, inplace=True)
        elif (
            ref_ranges[ref_ranges["label"] == col]["ref_range_upper"].isna().any()
            | ref_ranges[ref_ranges["label"] == col]["ref_range_lower"].isna().any()
        ):
            print("Ref range missing:", col)
            common_labs_missing_refs += [col]
        else:
            try:
                lower_bound = ref_ranges[ref_ranges["label"] == col]["ref_range_lower"]
                upper_bound = ref_ranges[ref_ranges["label"] == col]["ref_range_upper"]
                mean = (lower_bound + upper_bound) / 2
                std_dev = max(
                    ((upper_bound - lower_bound) / 3.92).values[0], 1e-6
                )  # set a minimum std_dev to avoid division by zero
                samples = np.random.normal(mean, std_dev, size=len(df[col]))
                df[col].fillna(pd.Series(samples), inplace=True)
            except IndexError:
                print("Error filling:", col)
    return df
