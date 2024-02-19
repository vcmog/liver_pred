from pathlib import Path

utils_dir = Path(__file__).parent.absolute()
project_dir = utils_dir.parent.parent.absolute()
data_dir = project_dir / "data"
sql_query_dir = utils_dir / "SQL queries"
output_dir = project_dir / "outputs"


# Propensity score matching options

perform_matching = False
no_matches = 5
