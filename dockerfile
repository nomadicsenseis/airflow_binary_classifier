FROM apache/airflow:2.5.3

USER airflow

# Install required packages for the sample DAG
RUN pip install numpy pandas tensorflow scikit-learn matplotlib seaborn xgboost lightgbm optuna

# Copy the DAG file into the dags directory
COPY regressor.py /opt/airflow/dags/


