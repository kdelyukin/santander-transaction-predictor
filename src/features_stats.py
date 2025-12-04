import pandas as pd
import numpy as np

def add_row_statistics(df, feature_cols):
    """
    Adds row-wise statistical features to the dataframe.
    """
    print("Generating row-wise statistics...")
    
    df['sum'] = df[feature_cols].sum(axis=1)
    df['mean'] = df[feature_cols].mean(axis=1)
    df['std'] = df[feature_cols].std(axis=1)
    df['min'] = df[feature_cols].min(axis=1)
    df['max'] = df[feature_cols].max(axis=1)
    df['skew'] = df[feature_cols].skew(axis=1)
    df['kurt'] = df[feature_cols].kurtosis(axis=1)
    
    return df