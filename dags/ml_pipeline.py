from airflow.models import DAG

from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.task_group import TaskGroup

from datetime import datetime

from utils.load_data import load_data
from utils.preprocess_data import preprocess_data
from utils.feature_selection import feature_selection
from utils.experiment import experiment
from utils.track_experiments_info import track_experiments_info
from utils.fit_best_model import fit_best_model
from utils.save_batch_data import save_batch_data
from utils.save_clean_batch_data import save_clean_batch_data
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
    "ml_pipeline",
    description='End-to-end ML pipeline example',
    schedule_interval='@daily',
    default_args=default_args, 
    catchup=False) as dag:

    
    # task: 1
    with TaskGroup('creating_storage_structures') as creating_storage_structures:

        # task: 1.1
        #creating_experiment_tracking_table = PostgresOperator(
         #   task_id="creating_experiment_tracking_table",
          #  postgres_conn_id='postgres_default',
           # sql='sql/create_experiments.sql'
        #)

        # task: 1.2
        creating_batch_data_table = PostgresOperator(
            task_id="creating_batch_data_table",
            postgres_conn_id='postgres_default',
            sql='sql/create_batch_data_table.sql'
        )

        # task: 1.3
        creating_clean_batch_data_table = PostgresOperator(
            task_id="creating_clean_batch_data_table",
            postgres_conn_id='postgres_default',
            sql='sql/create_clean_batch_data_table.sql'
        )

    # task: 2
    fetching_data = PythonOperator(
        task_id='fetching_data',
        python_callable=load_data

    )
    
    # task: 3
    with TaskGroup('preparing_data') as preparing_data:

        # task: 3.1
        preprocessing = PythonOperator(
            task_id='preprocessing',
            python_callable=preprocess_data
        )

        # task: 3.2
        saving_batch_data = PythonOperator(
            task_id='saving_batch_data',
            python_callable=save_batch_data
        )

        # task: 3.3
        #saving_clean_batch_data = PythonOperator(
         #   task_id='saving_clean_batch_data',
          #  python_callable=save_clean_batch_data
        #)

    # task: 4
    feature_selection = PythonOperator(
        task_id='feature_selection',
        python_callable=feature_selection
    )

    
        
    # task: 5
    hyperparam_tuning = PythonOperator(
        task_id='hyperparam_tuning',
        python_callable=experiment
    )

    # task: 6
    with TaskGroup('after_optuna') as after_optuna:

        # =======
        # task: 6.1        
        #saving_results = PythonOperator(
         #   task_id='saving_results',
          #  python_callable=track_experiments_info
        #)

        # task: 6.2
        fitting_best_model = PythonOperator(
            task_id='fitting_best_model',
            python_callable=fit_best_model
        )   

    #task 7
    selecting_features = PythonOperator(
        task_id='selecting_features',
        python_callable=select_features
    )
    #task 8
    making_predictions = PythonOperator(
        task_id='making_predictions',
        python_callable=make_predictions
    ) 

    #task 9
    evaluing_predictions = PythonOperator(
        task_id='evaluation',
        python_callable=eval_predictions
    )         

    creating_storage_structures >> fetching_data >> preparing_data >> feature_selection >> hyperparam_tuning >> after_optuna >> selecting_features >> making_predictions >> evaluing_predictions