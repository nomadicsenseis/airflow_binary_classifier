import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import PowerTransformer
from sklearn import model_selection
from utils.files_util import save_files, load_files

import utils.ml_pipeline_config as config

test_size = config.params["test_split_ratio"]

def preprocess_data():

    df = load_files(['df'])[0]
    #Drop correlated/non representative missing values
    df = df.dropna(subset=['rev_Mean','truck','avg6mou','change_mou','eqpdays'])
    #Use mean to fill in numerical non correlated/representative missing values
    num_cols = ['hnd_price','lor', 'adults', 'income', 'numbcars']  
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    #Use mode to fill in categorical non correlated/representative missing values
    cat_cols = ['area','prizm_social_one', 'hnd_webcap','ownrent','dwllsize','dwlltype','HHstatin','infobase'] 
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # Get a list of numeric column names
    numeric_cols = df.select_dtypes(include=np.number).drop('Customer_ID', axis=1).columns.tolist()

    # Define the proportion of negative values threshold
    neg_val_threshold = 0.1

    # Create a PowerTransformer instance
    pt = PowerTransformer()

    # Loop through each numeric column and check its skewness and neg vlaues
    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > 0.5:
          #Check how representative the negative values are.
          neg_val_prop = (df[col] < 0).sum() / len(df[col])
          if neg_val_prop > neg_val_threshold:
            # Apply the PowerTransformer to the column
            df[col] = pt.fit_transform(df[col].values.reshape(-1, 1)).flatten()
          else:
            # Apply log transformation to column
            df = df[df[col] >= 0]
            df[col] = np.log(df[col]+0.001)

    # Select the categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # One-hot encode the categorical columns
    df = pd.get_dummies(df, columns=cat_cols)

    df.name="clean_df"
    save_files([df])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(df.iloc[:,:-1], 
                                                                        df['churn'], 
                                                                        test_size=test_size)
    x_train.name = 'x_train'
    x_test.name = 'x_test'
    y_train.name = 'y_train'
    y_test.name = 'y_test'
    save_files([x_train, x_test, y_train, y_test])
