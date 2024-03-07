import os
import pandas as pd

pipeline_dir = os.path.dirname(__file__)
labels = pd.read_csv(
    os.path.join(pipeline_dir, "d_labitems.csv"),
    index_col="itemid",
    usecols=["itemid", "label", "fluid"],
)

labels["label"] = labels["fluid"] + " " + labels["label"]

UNIT_CONVERSIONS = [
    # (Name,                      Old Unit, Check function,   Conversion function)
    # TODO: add 'new unit' to this so function can update unit column
    # ('weight',                   'oz',    None,             lambda x: x/16.*0.45359237),   #to kg
    # ('weight',                   'lbs',   None,             lambda x: x*0.45359237),       #to kg
    # ('fraction inspired oxygen', None,    lambda x: x > 1,  lambda x: x/100.),             #to decimal
    # ('oxygen saturation',        None,    lambda x: x <= 1, lambda x: x*100.),             #to percentage
    # ('temperature',              'f',     lambda x: x > 79, lambda x: (x - 32) * 5./9),    #to C
    # ('height',                   'in',    None,             lambda x: x*2.54),             #to cm
    # ('50889',                    'mg/dL', None,             lambda x: x*10),               #C-Reactive Protein to mg/L
    # ('50916',                    'nG/mL', None,             lambda x: x/10),               #DHEA-Sulfate to ug/dl
    # ('50926',                    'mIU/L', None,             lambda x: x/1000),              #Follicle Stimulating Hormone to mIU/ml
    # ('50989',                    'ng/dL', None,             lambda x: x*10),                #Free testosterone to pg/ml
    #### MIMIC-IV SPECIFIC ####
    #'51464' Urine bilirubin, EU/dL to mg/dL is the same
    #'51654' HIV 1 viral load, log10 copies/mL both same units, abbreviated differently
    #'51282' Reticulocyte COunt, only 3 have /mm3 and they have no value, not worth converting
    #'51249' MCHC % and g/dL are the same measurement
    #'51229' Heparin LMW IU/ML and U/ML are the same
    # '51228' Heparin, ''
    # '50937 "Hepatitis A Virus Antibody" not actual units, N/A, null, pos/neg
    # 50918, double stranded dna, only one unit (one is null)
    # 51099 equivalent units
    # 50915 -ddimer  unclear if they're equivalent and no clear sanity check
    # 51085 - extra unit is just an n/a
    # 51196 -ddimer same as 50915
    # 50993 - equivalent
]


####TO DO: check returned datatypes
def process_lab_tests(
    labtests,
    threshold=0.8,
    save_to_csv=False,
    output_filename=None,
    output_path=None,
    compression=None,
    aggregate=False,
):
    """
    Process lab tests data by running a pipeline that includes clinical aggregation, missingness threshold, and unit conversion.

    Parameters:
    labtests (pd.DataFrame or str): The lab tests data. It can be a DataFrame or the path to the data file.
    threshold (float, optional): The missingness threshold. Only tests present in more than this percentage of patients are kept. Default is 0.8.
    save_to_csv (bool, optional): Whether to save the processed lab tests to a CSV file. Default is False.
    output_filename (str, optional): The name under which the processed lab tests will be saved. Required if save_to_csv is True.
    output_path (str, optional): The path where the processed lab tests file will be saved. Required if save_to_csv is True.
    compression (str, optional): The compression type for the output file. Can be "gzip". Default is None.
    aggregate (bool, optional): Whether to perform clinical aggregation. Default is False.

    Returns:
    pd.DataFrame: The processed lab tests data.

    """
    CURRENT_DIR = os.path.dirname(__file__)

    # load data if needed
    if isinstance(labtests, pd.DataFrame):
        lab_test_data = labtests
    else:
        if compression == "gzip":
            lab_test_data = pd.read_csv(
                labtests, compression="gzip"
            )  # compression='gzip'
        else:
            lab_test_data = pd.read_csv(labtests)

    # lab_data_filename = os.path.join(CURRENT_DIR, R'upper_GI_task', labtest_filename)
    # var_map_filename = os.path.join(CURRENT_DIR, 'labevent_to_variable_map.csv')
    ## TO DO: this needs to change
    if output_filename:
        output_filename = os.path.join(output_path, output_filename)

    lab_test_data = return_labels(lab_test_data)

    common_labs, lab_test_data = missingness_threshold(
        lab_test_data, threshold=threshold
    )
    # lab_test_data = standardize_units(lab_test_data)
    lab_test_data["index_date"] = pd.to_datetime(lab_test_data["index_date"])
    lab_test_data["charttime"] = pd.to_datetime(lab_test_data["charttime"])
    if save_to_csv == True:
        lab_test_data.to_csv(output_filename)
    return lab_test_data


def standardize_units(
    X, name_col="itemid", unit_col="valueuom", value_col="valuenum", inplace=True
):
    # takes input dataframe with name of measurement in 'name_col', current unit of measure in unit_col, and measured value in value_col.

    if not inplace:
        X = X.copy()
    name_col_vals = X[
        name_col
    ]  ##get_values_by_name_from_df_column_or_index(X, name_col)
    unit_col_vals = X[
        unit_col
    ]  ##get_values_by_name_from_df_column_or_index(X, unit_col)

    try:
        name_col_vals = name_col_vals.astype("string")
        unit_col_vals = unit_col_vals.astype("string")
    except:
        print("Can't call *.str")
        print(name_col_vals)
        print(unit_col_vals)
        raise

    # for each entry in UNIT_CONVERSIONS filter to find entries in dataframe with that measurement
    # print(name_col_vals, name_col_vals.dtype)
    name_filter = lambda n: name_col_vals.str.contains(n, case=False, na=False)
    unit_filter = lambda n: unit_col_vals.str.contains(n, case=False, na=False)

    for name, unit, rng_check_fn, convert_fn in UNIT_CONVERSIONS:
        name_filter_idx = name_filter(name)
        needs_conversion_filter_idx = (
            name_filter_idx & False
        )  # just creates a series of same dimensions but all false

        if unit is not None:
            needs_conversion_filter_idx |= name_filter(unit) | unit_filter(
                unit
            )  # default is false: if the unit is in the name or unit of the measurement, it is reassigned true
        if rng_check_fn is not None:
            needs_conversion_filter_idx |= rng_check_fn(X[value_col])  #

        idx = (
            name_filter_idx & needs_conversion_filter_idx
        )  # idx of all relevant values which need conversion

        X.loc[idx, value_col] = convert_fn(
            X[value_col][idx]
        )  # call conversion function

    return X


def subject_count(lab_df):
    # return number of unique subjects in df
    return lab_df["subject_id"].nunique()


def patients_per_variable(lab_df):
    # count of distinct patients which measurements for that variable
    patient_counts = lab_df.groupby("label").agg(
        count=("subject_id", pd.Series.nunique)
    )

    return patient_counts


def missingness_threshold(lab_df, threshold=0.8):
    ##remove entries where labs are present for less than the threshold of patients
    threshold_presence = threshold * subject_count(lab_df)
    counts = patients_per_variable(lab_df)
    labs_above_threshold = counts[counts["count"] > threshold_presence].index
    lab_df = lab_df[lab_df["label"].isin(labs_above_threshold)]
    if lab_df.empty:
        print("No data above threshold.")
    return labs_above_threshold, lab_df


def return_labels(lab_df):
    ## Takes itemid and returns the relevant label
    merged_df = lab_df.merge(labels, left_on="itemid", right_index=True, how="left")

    return merged_df
