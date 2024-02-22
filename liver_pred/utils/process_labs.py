import pandas as pd

# Import the "config" module
import config
from lab_pipeline.process_lab_tests import process_lab_tests

# Load the file "labs" from the "config.data_dir" directory
labs_file_path = config.data_dir / "interim/lab_events"
labs_data = pd.read_csv(labs_file_path, index_col=0)

# Now you can use the "labs_data" variable to work with the contents of the file
print(labs_data.head())

# Print the number of unique values in the "valueuom" column for each "itemid"
print(
    labs_data.groupby("itemid")["valueuom"]
    .nunique()
    .sort_values(ascending=False)
    .head(10)
)


processed_lab_data = process_lab_tests(
    labs_data, threshold=0.2, save_to_csv=False, aggregate=False
)

print(processed_lab_data.head())
