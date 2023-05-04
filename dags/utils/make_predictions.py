from utils.files_util import load_files, save_files
import pandas as pd
import os
import joblib
import glob


def make_predictions():
    #Loads cleaned/selected dataset to make predictions.
    x = load_files(['selected_x_val'])[0]

    #Loads latest model .pkl
    # Set the directory path where the .pkl files are saved
    dir_path = "/opt/airflow/models/"

    # Use glob to find all .pkl files in the directory
    file_list = glob.glob(os.path.join(dir_path, "*.pkl"))

    # Sort the file list by modified time in descending order
    file_list.sort(key=os.path.getmtime, reverse=True)

    # Load the latest .pkl file using joblib
    latest_model = joblib.load(file_list[0])

    #Makes prediction
    y_pred = latest_model.predict(x)
    y_pred_proba = latest_model.predict_proba(x)

    result_df = pd.DataFrame({
        'y_pred': y_pred,
        'y_pred_proba': [probas[1] if pred == 1 else probas[0] for pred, probas in zip(y_pred, y_pred_proba)]
    })

    result_df.name='predictions'

    save_files([result_df])
    


