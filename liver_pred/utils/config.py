from pathlib import Path

# Set pathnames
utils_dir = Path(__file__).parent.absolute()
project_dir = utils_dir.parent.parent.absolute()
data_dir = project_dir / "data"
sql_query_dir = utils_dir / "SQL queries"
output_dir = project_dir / "outputs"


# SQL options
sql_connection_str = "postgresql+psycopg2://postgres:postgres@localhost:5432/mimic4"

# Propensity score matching options

perform_matching = True
no_matches = 5


# Lab Processing Options
lab_threshold = 0.2
lab_aggregate = False
lab_save_to_csv = True
output_filename = "processed_lab_data.csv"
output_path = data_dir / "interim"


# Prediction model settings
model_order = 1
lead_time = 0
observation_window = None

# Window for 'current' lab tests in days around window
current_window = 1

# Window for binning historical lab tests in days
historical_window = 14
bin_window = 7

# Timepoints for proximal and distal features in days
proximal_timepoint = 180
distal_timepoint = 365
