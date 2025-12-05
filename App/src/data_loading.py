import pandas as pd
import os

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'train.csv')
    data_path = os.path.abspath(data_path)
    df = pd.read_csv(data_path)
    print(f"Data loaded from {data_path} \nwith shape {df.shape}")
    print(df.head())
    return df