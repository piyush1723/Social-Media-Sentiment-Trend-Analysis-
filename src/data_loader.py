import pandas as pd
import os

def load_data():
    base_path = os.path.dirname(os.path.dirname(__file__))  # project root
    file_path = os.path.join(base_path, "data", "raw", "twitter.csv")
    
    df = pd.read_csv(file_path)
    return df