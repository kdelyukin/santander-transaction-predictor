import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

def run_benchmark():
    print("Loading data...")
    df = pd.read_csv('data/train_magic.csv')
    
    # Define feature sets
    original_feats = [c for c in df.columns if 'var_' in c and 'count' not in c]
    magic_feats = [c for c in df.columns if 'count' in c]
    
    X = df.drop(['ID_code', 'target'], axis=1)
    y = df['target']
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Train on ORIGINAL features only
    clf_orig = GaussianNB()
    clf_orig.fit(X_train[original_feats], y_train)
    pred_orig = clf_orig.predict_proba(X_val[original_feats])[:, 1]
    auc_orig = roc_auc_score(y_val, pred_orig)
    print(f"Baseline AUC (Original Features): {auc_orig:.4f}")
    
    # 2. Train on MAGIC features only
    clf_magic = GaussianNB()
    clf_magic.fit(X_train[magic_feats], y_train)
    pred_magic = clf_magic.predict_proba(X_val[magic_feats])[:, 1]
    auc_magic = roc_auc_score(y_val, pred_magic)
    print(f"Magic AUC (Count Features):     {auc_magic:.4f}")
    
    print("-" * 30)
    print(f"Improvement: {auc_magic - auc_orig:.4f}")

if __name__ == "__main__":
    run_benchmark()