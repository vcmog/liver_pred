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

perform_matching = False
no_matches = 5
