from utils.files_util import save_files, load_files
import pandas as pd
from sklearn.metrics import accuracy_score

def eval_predictions():
    df_y_pred =load_files(['predictions'])[0]
    y_pred = df_y_pred['y_pred']

    y_val = load_files(['y_val'])[0]

    accuracy = accuracy_score(y_pred,y_val)

    acc=pd.DataFrame({'accuracy': [accuracy]})

    acc.name='accuracy'

    save_files([acc])


