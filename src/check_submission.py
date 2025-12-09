import pandas as pd
import numpy as np
import os

def check_submission(
    submission_path="submission.csv",
    test_path="data/test.csv"
):
    print("🔍 Checking submission file...\n")

    # --- Check 1: File existence ---
    if not os.path.exists(submission_path):
        print(f"❌ File not found: {submission_path}")
        return
    if not os.path.exists(test_path):
        print(f"⚠️ Test file not found: {test_path}")
        print("    Skipping ID alignment check.\n")
        test = None
    else:
        test = pd.read_csv(test_path)

    sub = pd.read_csv(submission_path)
    print(f"✅ Loaded {submission_path} ({sub.shape[0]} rows, {sub.shape[1]} cols)")

    # --- Check 2: Columns ---
    expected_cols = ["ID_code", "target"]
    if list(sub.columns) != expected_cols:
        print(f"❌ Column mismatch! Expected {expected_cols}, got {list(sub.columns)}")
        return
    print("✅ Columns OK")

    # --- Check 3: Row count ---
    if test is not None:
        if sub.shape[0] != test.shape[0]:
            print(f"❌ Row count mismatch! submission={sub.shape[0]}, test={test.shape[0]}")
            return
        print("✅ Row count matches test.csv")

    # --- Check 4: ID alignment ---
    if test is not None:
        if not (sub["ID_code"].equals(test["ID_code"])):
            print("❌ ID_code order or values do not match test.csv!")
            mismatches = (sub["ID_code"] != test["ID_code"]).sum()
            print(f"    {mismatches} mismatched IDs found.")
            return
        print("✅ ID_code alignment verified")

    # --- Check 5: Target sanity ---
    tmin, tmax = sub["target"].min(), sub["target"].max()
    if (tmin < 0) or (tmax > 1):
        print(f"❌ Target values out of [0,1] range (min={tmin:.4f}, max={tmax:.4f})")
        return
    print(f"✅ Target range OK (min={tmin:.4f}, max={tmax:.4f})")

    if sub["target"].isnull().any():
        print("❌ Missing values detected in target column")
        return
    print("D No missing values")

    # --- Check 6: Distribution sanity ---
    desc = sub["target"].describe()
    print("\nTarget distribution:")
    print(desc.to_string())

    mean_val = desc["mean"]
    if mean_val > 0.4 or mean_val < 0.01:
        print("⚠️  Unusual mean target value")
    else:
        print("✅ Mean target acceptable")

    print("\n✅ All checks complete")


if __name__ == "__main__":
    check_submission()
