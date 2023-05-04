from airflow.models import DAG

from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

from datetime import datetime

from utils.load_data import load_data
from utils.preprocess_data import preprocess_data
from utils.select_features import select_features
from utils.make_predictions import make_predictions
from utils.eval_predictions import eval_predictions

default_args= {
    'owner': 'Diego Rivera',
    'email_on_failure': False,
    'email': ['nomadicsenseis@gmail.com'],
    'start_date': datetime(2021, 12, 1)
}

with DAG(
    "prediction_pipeline",
    description='Prediction ML pipeline example',
    schedule_interval='@daily',
    default_args=default_args, 
    catchup=False) as dag:

    
    # task: 1
    #Creating data structures in PostgresSQL


    # task: 2
    #fetching_data
    fetching_data = PythonOperator(
        task_id='fetching_data',
        python_callable=load_data
    ) 

    preprocessing = PythonOperator(
        task_id='preprocessing',
        python_callable=preprocess_data
    )

    selecting_features = PythonOperator(
        task_id='selecting_features',
        python_callable=select_features
    )

    making_predictions = PythonOperator(
        task_id='making_predictions',
        python_callable=make_predictions
    ) 

    evaluing_predictions = PythonOperator(
        task_id='evaluation',
        python_callable=eval_predictions
    )         

    fetching_data >> preprocessing  >> selecting_features >> making_predictions >> evaluing_predictions
