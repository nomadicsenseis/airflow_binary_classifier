params = {
    "db_engine": "postgresql+psycopg2://airflow:airflow@postgres/airflow",
    "db_schema": "public",
    "db_experiments_table": "experiments",
    "db_batch_table": "batch_data",
    "db_clean_batch_table": "clean_batch_data",
    "test_split_ratio": 0.3,
    "grid_cv_folds": 3,
    "num_sel_features": 10
}   