# imputation check 
# encoding check
# feature choosing 
# storing inference of function
# storing jason of the feature 
# split the data
# return

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split

def check_impute(df):
    print(f"columns : {df.columns}")
    print(f"shape before imputation: {df.shape}")
    print(f"missing values before imputation: \n{df.isnull().sum()}")
    for i in df.columns:
        if df[i].dtype != 'object':
            if df[i].isnull().sum()/len(df) < 0.30 and df[i].isnull().sum()!=0:
                high_mod = df[i].mode()[0]
                df[i].fillna(high_mod, inplace=True)
            elif df[i].isnull().sum()==0:
                print(f"column {i} has no missing value")
            else:
                print(f"column {i} has more than 30% missing value")
                df = df.drop(columns=[i])
                print(f"column {i} dropped")
        else:
            # check for the missing more than 30%
            if df[i].isnull().sum()/len(df) == 0:
                print(f"column {i} has no missing value")
            elif df[i].isnull().sum()/len(df) < 0.30:
                # check for the skewedness
                skew_val = df[i].skew()
                if abs(skew_val) > 0.5:
                    # impute with median
                    median_val = df[i].median()
                    df[i].fillna(median_val, inplace=True)
                    print(f"column {i} is skewed with skewness {skew_val}, imputed with median")
                else:
                    # impute with mode
                    mean_val = df[i].mean()[0]
                    df[i].fillna(mean_val, inplace=True)
                    print(f"column {i} is not skewed with skewness {skew_val}, imputed with mean")
    
    print(f"missing values after imputation: \n{df.isnull().sum()}")
    print(f"shape after imputation: {df.shape}")
    print("imputation completed")
    return df

def cat_encoding(df):
    cat_val = ["Type"]
    for i in cat_val:
        df['Type'] = df['Type'].map({'L':1, 'M':2, 'H':3})
    print("categorical encoding completed")
    path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'cat_encoder.joblib')
    path = os.path.abspath(path)
    joblib.dump(cat_val, path)
    print(f"categorical encoder saved at {path}")
    return df


def preprocess_data(df):
    data = df.copy()
    data.columns = data.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    data = check_impute(data)
    data = cat_encoding(data)
    data = data.drop(columns=['Air_temperature_K_','Product_ID','id'])
    print(data.describe().T)
    print(f"final columns choosen : {data.columns}")
    # store in txt file  about the final features
    path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'final_features.txt')
    path = os.path.abspath(path)
    with open(path, 'w') as f:
        j = 0
        for col in data.columns:
            f.write(f"{j}. {col}\n")
            j += 1
    print(f"final features saved at {path}")
    X = data.drop(columns=['Machine_failure'])
    Y = data['Machine_failure']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
    print(f"Preprocessing completed. \nX_train shape: {X_train.shape}, Y_train shape: {Y_train.shape} \nX_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    return X_train, X_test, Y_train, Y_test
