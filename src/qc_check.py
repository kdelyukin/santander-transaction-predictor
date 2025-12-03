import pandas as pd
import numpy as np

def check_quality():
    print("Checking quality of Data...")
    df = pd.read_csv('data/train_magic.csv')

    # 1. Check for NaNs
    na_count = df.isna().sum().sum()
    print(f"Total NaNs in dataset: {na_count} (Should be 0)")

    # 2. Check for Infinite values 
    inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
    print(f"Total Infinite values: {inf_count} (Should be 0)")

    # 3. Check shape
    print(f"Shape: {df.shape} (Should have 200 original + 200 magic cols)")

if __name__ == "__main__":
    check_quality()