import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from .validation import prepare_data, get_stratified_folds, evaluate_oof_auc
from .data_augmentation import oversample_minority


def train_lgbm_cv(
    n_splits: int = 5,
    use_stats: bool = True,
    use_augmentation: bool = False
):
    """
    Trains LightGBM with Stratified K-Fold CV.
    Saves test predictions to outputs/lgbm_predictions.csv.
    """
    print("Preparing data for LightGBM...")
    X, y, X_test, test_ids = prepare_data(add_stats=use_stats)

    folds = get_stratified_folds(y, n_splits=n_splits)
    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    os.makedirs("outputs", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n===== LightGBM Fold {fold}/{n_splits} =====")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if use_augmentation:
            X_train, y_train = oversample_minority(X_train, y_train)

        model = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        oof_preds[val_idx] = val_pred

        print(f"Fold {fold} AUC: {fold_auc:.5f}")

        test_pred = model.predict_proba(X_test)[:, 1]
        test_preds += test_pred / n_splits

    oof_auc = evaluate_oof_auc(y, oof_preds)
    print(f"\nLightGBM OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUCs: {[round(s, 5) for s in fold_scores]}")

    out_path = "outputs/lgbm_predictions.csv"
    pd.DataFrame({"ID_code": test_ids, "target": test_preds}).to_csv(out_path, index=False)
    print(f"Saved LightGBM test predictions to {out_path}")

    return fold_scores, oof_auc


if __name__ == "__main__":
    train_lgbm_cv()
