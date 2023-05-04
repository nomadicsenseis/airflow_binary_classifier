import pandas as pd
from datetime import datetime
import joblib

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from utils.files_util import load_files, save_files

def fit_best_model():
    x_train, x_val, y_train, y_val, best_params = load_files(['x_train', 'x_val', 'y_train', 'y_val', 'exp_info'])
    model = XGBClassifier(max_depth=best_params['best_xgb_max_depth'].values[0],
                          n_estimators=best_params['best_xgb_n_estimators'].values[0],
                          min_child_weight=best_params['best_xgb_min_child_weight'].values[0],
                          gamma=best_params['best_xgb_gamma'].values[0],
                          learning_rate=best_params['best_xgb_learning_rate'].values[0],
                          subsample=best_params['best_xgb_subsample'].values[0],
                          colsample_bytree=best_params['best_xgb_colsample_bytree'].values[0],
                          reg_alpha=best_params['best_xgb_reg_alpha'].values[0],
                          reg_lambda=best_params['best_xgb_reg_lambda'].values[0],
                          tree_method= 'auto',  # Use GPU for faster training if available
                          objective= 'binary:logistic',
                          eval_metric= 'auc')
        
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracy_best_model=pd.DataFrame.from_dict({'accuracy_best_model':[accuracy]})
    accuracy_best_model.name='accuracy_saved_model'
    save_files([accuracy_best_model])

    # save best model
    now = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    filename = 'model_' + now + '.pkl'
    joblib.dump(model, '/opt/airflow/models/' + filename, compress=1)
