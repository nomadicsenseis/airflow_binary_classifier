import pandas as pd
from datetime import datetime
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from utils.files_util import save_files, load_files

def feature_selection():
  x_train, x_test, y_train, y_test = load_files(['x_train', 'x_test', 'y_train', 'y_test'])

  # Create logistic regression model with L1 regularization
  logreg = LogisticRegressionCV(penalty='l1', solver='saga', cv=2)
  logreg.fit(x_train, y_train)
  
  k=50
  # get the absolute values of the coefficients
  coef_abs = np.abs(logreg.coef_)
  #Get the top_k indexes
  topk_indices = np.argpartition(coef_abs.flatten(), -k)[-k:]
  # get the corresponding feature names
  feature_names = list(x_train.columns)
  topk_features = [feature_names[i] for i in topk_indices]
  df_topk_features = pd.DataFrame(topk_features, columns=['selected_features'])
  df_topk_features.name='selected_features'
  save_files([df_topk_features])

  x_train=x_train[topk_features]
  x_test=x_test[topk_features]

  x_train.name = 'selected_x_train'
  x_test.name = 'selected_x_test'
  y_train.name = 'selected_y_train'
  y_test.name = 'selected_y_test'
  
  save_files([x_train, x_test, y_train, y_test])