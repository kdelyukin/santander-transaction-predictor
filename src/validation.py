import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from .data_loader import load_data
from .features_stats import add_row_statistics


def prepare_data(add_stats: bool = True):
    """
    Loads train/test and returns:
    X (train features), y (train target), X_test, test_ids.
    """
    train, test = load_data()  # uses var* downcasting already :contentReference[oaicite:0]{index=0}

    feature_cols = [c for c in train.columns if "var" in c]

    if add_stats:
        train = add_row_statistics(train, feature_cols)
        test = add_row_statistics(test, feature_cols)  # row-wise stats :contentReference[oaicite:1]{index=1}

    X = train.drop(columns=["ID_code", "target"])
    y = train["target"]
    X_test = test.drop(columns=["ID_code"])
    test_ids = test["ID_code"]

    return X, y, X_test, test_ids


def get_stratified_folds(y, n_splits: int = 5, random_state: int = 42, shuffle: bool = True):
    """
    Returns list of (train_idx, val_idx) for Stratified K-Fold.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )
    return list(skf.split(np.zeros(len(y)), y))


def evaluate_oof_auc(y_true, oof_preds):
    """
    Computes overall AUC from out-of-fold predictions.
    """
    return roc_auc_score(y_true, oof_preds)
