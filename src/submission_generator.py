import os
import pandas as pd


def load_prediction_file(path, name):
    """
    Loads a prediction file with columns: ID_code, target.
    Renames 'target' to f'target_{name}'.
    """
    df = pd.read_csv(path)
    df = df[["ID_code", "target"]].copy()
    df.rename(columns={"target": f"target_{name}"}, inplace=True)
    return df


def generate_ensemble_submission(
    output_path: str = "outputs/submission_ensemble.csv"
):
    """
    Loads available model prediction files and creates an averaged ensemble.
    """
    os.makedirs("outputs", exist_ok=True)

    candidates = [
        ("outputs/lgbm_predictions.csv", "lgbm", 1.0),
        ("outputs/xgb_predictions.csv", "xgb", 1.0),
    ]

    dfs = []
    weights = []

    for path, name, w in candidates:
        if os.path.exists(path):
            print(f"Using predictions from: {path}")
            dfs.append(load_prediction_file(path, name))
            weights.append(w)
        else:
            print(f"Warning: {path} not found; skipping.")

    if not dfs:
        raise FileNotFoundError("No prediction files found to ensemble.")

    # Merge all on ID_code
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="ID_code", how="inner")

    weight_sum = sum(weights)
    pred_cols = [c for c in merged.columns if c.startswith("target_")]

    # Simple weighted average
    merged["target"] = 0.0
    for (path, name, w) in candidates:
        col = f"target_{name}"
        if col in merged.columns:
            merged["target"] += w * merged[col] / weight_sum

    submission = merged[["ID_code", "target"]]
    submission.to_csv(output_path, index=False)
    print(f"Saved ensemble submission to {output_path}")

    root_path = "submission.csv"
    submission.to_csv(root_path, index=False)
    print(f"Saved Kaggle submission file to {root_path}")


if __name__ == "__main__":
    generate_ensemble_submission()
