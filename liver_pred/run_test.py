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

current_labs = feature_generation.lab_within_n_days(
    processed_labs, config.current_window
)
current_labs = md.fill_na_midnormal(current_labs)
historical_labs = feature_generation.historical_labs(
    processed_labs, config.historical_window
)


feature_generation.write_report(current_labs, historical_labs, config.output_dir)

binned_labs = feature_generation.bin_measurements(historical_labs)

trend_features = feature_generation.generate_trend_features(
    binned_labs, current_labs, config.proximal_timepoint, config.distal_timepoint
)
print(trend_features.head())
