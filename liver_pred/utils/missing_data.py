import pandas as pd

ref_ranges = {
    "Albumin": {"lower": 3.4, "upper": 5.2},  # itemid=50862
    "Basophils": {"lower": 0, "upper": 2},  # itemid=51146
    "Eosinophils": {"lower": 0, "upper": 7},  # itemid=51200
    "Neutrophils": {"lower": 34, "upper": 71},  # itemid=51256
    "Monocytes": {"lower": 2, "upper": 13},  # itemid=51254
    "Lymphocytes": {"lower": 18, "upper": 53},  # itemid=51254
    "PTT": {"lower": 22, "upper": 36.5},  # itemid=51275
    "Alkaline Phosphatase": {"lower": 35, "upper": 130},  # itemid=50863
    "Bilirubin, Total": {"lower": 0, "upper": 1.5},  # itemid=50885
    "Alanine Aminotransferase (ALT)": {"lower": 0, "upper": 40},  # itemid=50861
    "Asparate Aminotransferase (AST)": {"lower": 0, "upper": 40},  # itemid=50878
    "INR(PT)": {"lower": 0.9, "upper": 1.1},  # itemid=51237
    "PT": {"lower": 9.4, "upper": 13.4},  # itemid=51274
    "Phosphate": {"lower": 2.7, "upper": 4.5},  # itemid=50970
    "Calcium, Total": {"lower": 8.4, "upper": 10.3},  # itemid=50893
    "Magnesium": {"lower": 1.6, "upper": 2.6},  # itemid=50960
    "Glucose": {
        "lower": 70,
        "upper": 105,
    },  # itemid=50931 ###URGENT: CURRENT CONTAINS URINE TOO###
    "Bicarbonate": {"lower": 22, "upper": 32},  # itemid=50882
    "Anion Gap": {
        "lower": 8,
        "upper": 20,
    },  # itemid=50868,52500 ###URGENT: why do I have none of these?
    "Red Blood Cells": {"lower": 3.9, "upper": 6.2},  # itemid=51279
    "RDW": {"lower": 10.5, "upper": 15.5},  # itemid=51277
    "MCH": {"lower": 26, "upper": 32},  # itemid=51248
    "Platelet Count": {"lower": 150, "upper": 440},  # itemid=51265
    "MCV": {"lower": 82, "upper": 98},  # itemid=51250
    "MCHC": {"lower": 31, "upper": 37},  # itemid=51249
    "Hemoglobin": {"lower": 12, "upper": 18},  # itemid=51222,50811
    "White Blood Cells": {"lower": 4, "upper": 11},  # itemid=51301,51755
    "Chloride": {"lower": 96, "upper": 108},  # itemid=50902,52535
    "Sodium": {"lower": 133, "upper": 147},  # itemid=52623,50983
    "Urea Nitrogen": {"lower": 6, "upper": 20},  # itemid=51006,52647
    "Potassium": {"lower": 3.3, "upper": 5.1},  # itemid=52610,50971
    "Creatinine": {"lower": 0.4, "upper": 1.2},  # itemid=50912,52546
    "Hematocrit": {"lower": 34, "upper": 52},  # itemid=51221
}


def fill_na_midnormal(df):
    for test in ref_ranges:
        ref_ranges[test]["centre"] = round(
            (ref_ranges[test]["upper"] + ref_ranges[test]["lower"]) / 2, 2
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
            df[col].fillna(inplace=True)
    return df
