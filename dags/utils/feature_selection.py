import pandas as pd
from datetime import datetime
import numpy as np


from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from utils.files_util import save_files, load_files

def feature_selection():
  df = load_files(['clean_df'])[0]

  x_train, x_test, y_train, y_test = model_selection.train_test_split(df.iloc[:,:-1], 
                                                                        df['churn'], 
                                                                        test_size=0.3, random_state=42)

  # Create logistic regression model with L1 regularization
  logreg = LogisticRegressionCV(penalty='l1', solver='saga', cv=5)
  logreg.fit(x_train, y_train)
  
  k=10
  # get the absolute values of the coefficients
  coef_abs = np.abs(logreg.coef_)
  #Get the top_k indexes
  topk_indices = np.argpartition(coef_abs.flatten(), -k)[-k:]
  # get the corresponding feature names
  feature_names = list(x_train.columns)
  topk_features = [feature_names[i] for i in topk_indices]

  #Save selected df
  selected_df=df[topk_features+['Customer_ID','churn']]

  # Split df into 80% train and 20% test sets
  df_train, df_test = model_selection.train_test_split(selected_df, test_size=0.2, random_state=42)
  df_train.name='selected_df_train_val'
  df_test.name='selected_df_test'
  save_files([df_train,df_test])


