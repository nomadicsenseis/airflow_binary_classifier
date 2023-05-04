import numpy as np
import pandas as pd
from datetime import datetime
import optuna

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from utils.files_util import save_files, load_files
import utils.ml_pipeline_config as config

def objective(trial):
    # Define the hyperparameters to be tuned
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'tree_method': 'auto',  # Use GPU for faster training if available
        'objective': 'binary:logistic',
        'eval_metric': 'auc'
    }
    x_train,y_train,x_val,y_val=load_files(['x_train','y_train','x_val','y_val'])
    # Train the model
    model = XGBClassifier(**params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=10, verbose=False)
    # Calculate the validation AUC score
    auc = model.evals_result()['validation_0']['auc'][-1]
    # Return the AUC score as the objective value to be maximized
    return auc


def experiment():

    df = load_files(['selected_df_train_val'])[0]

    x_train, x_val, y_train, y_val = model_selection.train_test_split(df.iloc[:,:-1], 
                                                                        df['churn'], 
                                                                        test_size=0.3, random_state=42)

    x_train.name='x_train'
    x_val.name='x_val'
    y_train.name='y_train'
    y_val.name='y_val'

    save_files([x_train,x_val,y_train,y_val])

    # Set up the study
    study = optuna.create_study(direction='maximize')

    # Optimize the hyperparameters
    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters and corresponding AUC score
    best_params = study.best_params
    best_xgb_max_depth=best_params["max_depth"]	
    best_xgb_n_estimators=best_params["n_estimators"]
    best_xgb_min_child_weight=best_params["min_child_weight"]
    best_xgb_gamma=best_params["gamma"]
    best_xgb_reg_alpha=best_params["reg_alpha"]
    best_xgb_reg_lambda=best_params["reg_lambda"]
    best_xgb_learning_rate=best_params["learning_rate"]
    best_xgb_subsample=best_params["subsample"]
    best_xgb_colsample_bytree=best_params["colsample_bytree"]
    best_auc = study.best_value
    

    # save esperiments information for historical persistence
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    exp_info = pd.DataFrame([[now,
                              best_auc,
                              best_xgb_n_estimators,
                              best_xgb_max_depth,
                              best_xgb_min_child_weight,
                              best_xgb_gamma,
                              best_xgb_learning_rate,
                              best_xgb_subsample,
                              best_xgb_colsample_bytree,
                              best_xgb_reg_alpha,
                              best_xgb_reg_lambda]],
                            columns=['experiment_datetime',
                                     'best_auc',
                                     'best_xgb_n_estimators',
                                     'best_xgb_max_depth',
                                     'best_xgb_min_child_weight',
                                     'best_xgb_gamma',
                                     'best_xgb_learning_rate',
                                     'best_xgb_subsample',
                                     'best_xgb_colsample_bytree',
                                     'best_xgb_reg_alpha',
                                     'best_xgb_reg_lambda'
                                    ])
    exp_info.name = 'exp_info'

    save_files([exp_info])