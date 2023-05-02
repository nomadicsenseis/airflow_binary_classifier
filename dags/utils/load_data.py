import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from utils.files_util import save_files

def load_data():
    
    df=pd.read_csv('/opt/airflow/data/dataset.csv', delimiter=";",decimal=",")
    df.name="df"
    save_files([df])