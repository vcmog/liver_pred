import pandas as pd

# Import the "config" module
import config as config
from lab_pipeline.process_lab_tests import process_lab_tests

# Load the file "labs" from the "config.data_dir" directory
labs_file_path = config.data_dir / "interim/lab_events"
labs_data = pd.read_csv(labs_file_path, index_col=0)
cohort_ids = pd.read_csv(
    config.data_dir / "interim/matched_cohort_ids.csv", index_col=0
)
# Now you can use the "labs_data" variable to work with the contents of the file

# Print the number of unique values in the "valueuom" column for each "itemid"
# print(
#    labs_data.groupby("itemid")["valueuom"]
#    .nunique()
#    .sort_values(ascending=False)
#    .head(10)
# )

print("Number of subject_ids before processing:", labs_data["subject_id"].nunique())
processed_lab_data = process_lab_tests(
    labs_data,
    threshold=config.lab_threshold,
    save_to_csv=config.lab_save_to_csv,
    aggregate=config.lab_aggregate,
    output_filename=config.output_filename,
    output_path=config.output_path,
)


subjects = processed_lab_data["subject_id"].nunique()
n_cases = sum(
    cohort_ids[cohort_ids["subject_id"].isin(processed_lab_data["subject_id"])][
        "outcome"
    ]
)
print("Number of subject_ids after processing:", subjects)
print("Number of cases after processing:", n_cases, "(", n_cases / subjects * 100, "%)")
print("Processed lab data:")
print(processed_lab_data.head())
print("Number of Unique Variables pre-processing:")
print(labs_data["itemid"].nunique())
print("Number of Unique Variables post-processing:")
print(processed_lab_data["itemid"].nunique())
