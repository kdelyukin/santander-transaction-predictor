import pandas as pd
import numpy as np
from tqdm import tqdm

def get_real_test_indices(test_df):
    """
    Identifies 'Real' rows in the test set. 
    Strategy: Real rows usually contain at least one unique value 
    across the 200 features relative to the test set itself.
    """
    print("Identifying real test rows...")
    # Calculate value counts for every column in the test set
    unique_count = np.zeros_like(test_df.iloc[:, 1:].values)
    
    for i, col in enumerate(tqdm(test_df.columns[1:])):  # Skip ID_code
        # Map counts to the test data
        counts = test_df[col].map(test_df[col].value_counts())
        # Mark locations where the value appears only once (unique)
        unique_count[:, i] = (counts == 1).values
        
    # A row is REAL if it has at least one unique value
    real_indices = np.argwhere(np.sum(unique_count, axis=1) > 0).flatten()
    return real_indices

def generate_magic_features(train, test):
    """
    Generates frequency count features (var_0_count, var_1_count, etc.)
    CRITICAL: Only use Train + Real Test rows for counting!
    """
    print("Generating Magic Features...")
    
    # 1. Identify Real Test Rows
    real_test_idxs = get_real_test_indices(test)
    real_test = test.iloc[real_test_idxs]
    
    # 2. Combine Train + Real Test for accurate frequency counting
    combined_df = pd.concat([train, real_test], axis=0)
    
    # 3. Create Count Features
    features = [c for c in train.columns if c not in ['ID_code', 'target']]
    
    for col in tqdm(features):
        # Calculate frequency on the CLEAN set (Train + Real Test)
        count_map = combined_df[col].value_counts().to_dict()
        
        # Map these counts back to the original Train and Test DFs
        train[f'{col}_count'] = train[col].map(count_map)
        test[f'{col}_count'] = test[col].map(count_map)
        
    return train, test

if __name__ == "__main__":
    # Test the logic
    from data_loader import load_data
    tr, te = load_data()
    tr_magic, te_magic = generate_magic_features(tr, te)
    
    print("Magic features created!")
    print(tr_magic[['var_0', 'var_0_count']].head())
    
    tr_magic.to_csv('data/train_magic.csv', index=False)
    te_magic.to_csv('data/test_magic.csv', index=False)