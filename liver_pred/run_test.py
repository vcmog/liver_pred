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

cur, trend, feature_df = feature_generation.generate_features(
    processed_labs,
    cohort_ids,
    config.current_window_preindex,
    config.current_window_postindex,
    config.historical_window,
)
feature_df.to_csv(data_dir / "interim/feature_df.csv")
print("ColonFlag style feature generation complete")

threed = feature_generation.create_array_for_CNN(processed_labs, -3, 150)
print("3D array generation complete")
