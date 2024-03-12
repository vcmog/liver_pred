import pandas as pd
import numpy as np
import config

ref_ranges = pd.read_csv(config.utils_dir / "ref_ranges.csv")
ref_ranges["label"] = ref_ranges["fluid"] + " " + ref_ranges["label"]
ref_ranges["centre"] = round(
    (ref_ranges["ref_range_upper"] + ref_ranges["ref_range_lower"]) / 2, 2
)
non_ref_ranges = ref_ranges[
    ref_ranges["ref_range_upper"].isna() | ref_ranges["ref_range_lower"].isna()
]


def fill_na_midrange(df):
    for test in ref_ranges:
        ref_ranges[test]["centre"] = round(
            (ref_ranges[test]["ref_range_upper"] + ref_ranges[test]["ref_range_lower"])
            / 2,
            2,
        )
    for col in df.columns:
        if "trend" in col:
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna(ref_ranges[col]["centre"], inplace=True)
    return df


def fill_nas_mean(df):
    for col in df.columns:
        if "trend" in col:
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna(df["col"].mean(skipna=True), inplace=True)
    return df


def fill_nas_normal(df):
    for col in df.columns:
        if "trend" in col:
            df[col].fillna(0, inplace=True)
        else:
            lower_bound = ref_ranges[col]["ref_range_lower"]
            upper_bound = ref_ranges[col]["ref_range_upper"]
            mean = (lower_bound + upper_bound) / 2
            std_dev = (upper_bound - lower_bound) / 3.92
            samples = np.random.normal(mean, std_dev, num_samples=len(df[col]))
            df[col].fillna(samples, inplace=True)
    return df
