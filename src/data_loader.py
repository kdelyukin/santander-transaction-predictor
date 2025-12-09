import pandas as pd
import numpy as np

def load_data(data_path='data/'):
    """
    Loads train and test data with memory optimization.
    """
    print("Loading data...")
    train = pd.read_csv(f'{data_path}train_magic.csv')
    test = pd.read_csv(f'{data_path}test_magic.csv')

    print("Optimizing memory...")
    # Downcast float64 to float32 to save memory
    for col in train.columns:
        if 'var' in col:
            train[col] = train[col].astype(np.float32)
            test[col] = test[col].astype(np.float32)
            
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test

if __name__ == "__main__":
    load_data()