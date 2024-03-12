import utils.config as config
import pandas as pd
import utils.colonflag.feature_generation as feature_generation
import datetime as dt
import utils.missing_data as md

data_dir = config.data_dir

processed_labs = pd.read_csv(
    data_dir / "interim/processed_lab_data.csv",
    parse_dates=["index_date", "charttime"],
    date_format="%Y-%m-%d %H:%M:%S",
    index_col=0,
)
cohort_ids = pd.read_csv(data_dir / "interim/matched_cohort_ids.csv", index_col=0)
# Create a dataframe containing the current lab measurements
current_labs = feature_generation.current_bloods_df(
    processed_labs, config.current_window
)

outcome = current_labs["outcome"]

# Fill missing values in the current_labs dataframe
current_labs = md.fill_nas_normal(current_labs.drop("outcome", axis=1))

current_labs["outcome"] = outcome  # add outcome back to the dataframe
# Form historical df
historical_labs = feature_generation.historical_labs(
    processed_labs, config.historical_window
)

current_labs, historical_labs, removed = md.remove_if_missing_from_other(
    current_labs, historical_labs
)

historical_labs.to_csv(config.utils_dir / "historical_labs.csv")

feature_generation.write_report(current_labs, historical_labs, config.output_dir)
# Remove subject_ids from current_labs and historical_labs that are not present in both

binned_labs = feature_generation.bin_measurements(historical_labs)

current_labs.to_csv(config.utils_dir / "current_labs.csv")
binned_labs.to_csv(config.utils_dir / "binned_labs.csv")
trend_features = feature_generation.generate_trend_features(
    binned_labs, current_labs, config.proximal_timepoint, config.distal_timepoint
)
