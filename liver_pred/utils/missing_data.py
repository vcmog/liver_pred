import pandas as pd
import numpy as np
import utils.config as config

if not config.utils_dir / "ref_ranges.csv":
    print("No reference range file found, upload using SQL query")
ref_ranges = pd.read_csv(config.utils_dir / "ref_ranges.csv")


def process_ref_ranges(ref_range_csv):
    if ref_ranges.columns.contains("centre"):
        return ref_range_csv
    ref_ranges["label"] = ref_ranges["fluid"] + " " + ref_ranges["label"]
    ref_ranges = (
        ref_ranges.groupby("label")
        .agg({"ref_range_lower": "min", "ref_range_upper": "max"})
        .reset_index()
    )
    ref_ranges["centre"] = round(
        (ref_ranges["ref_range_upper"] + ref_ranges["ref_range_lower"]) / 2, 2
    )
    pd.save_csv(config.utils_dir / "ref_ranges.csv", ref_ranges)
    return ref_ranges


ref_ranges = process_ref_ranges(ref_ranges)


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
            ref_ranges[ref_ranges["label"] == col]["ref_range_upper"].isna()
            | ref_ranges[ref_ranges["label"] == col]["ref_range_lower"].isna()
        ):
            print("Ref range missing:", col)
            common_labs_missing_refs += [col]
        else:
            lower_bound = ref_ranges[ref_ranges["label"] == col]["ref_range_lower"]
            upper_bound = ref_ranges[ref_ranges["label"] == col]["ref_range_upper"]
            mean = (lower_bound + upper_bound) / 2
            std_dev = (upper_bound - lower_bound) / 3.92
            samples = np.random.normal(mean, std_dev, num_samples=len(df[col]))
            df[col].fillna(samples, inplace=True)
    return df
