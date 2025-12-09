import numpy as np
import pandas as pd


def oversample_minority(X, y, multiplier: int = 2, random_state: int = 42):
    """
    Simple oversampling of the minority (target == 1) class.
    multiplier=2 → add 2x more minority samples (approx).
    """
    y = pd.Series(y).reset_index(drop=True)
    X = pd.DataFrame(X).reset_index(drop=True)

    minority_mask = y == 1
    X_min = X[minority_mask]
    y_min = y[minority_mask]

    if X_min.empty:
        print("No minority samples found; skipping augmentation.")
        return X, y

    X_list = [X]
    y_list = [y]

    rng = np.random.RandomState(random_state)

    for i in range(multiplier):
        X_sampled = X_min.sample(
            frac=1.0,
            replace=True,
            random_state=rng.randint(0, 1_000_000)
        )
        y_sampled = pd.Series(1, index=X_sampled.index)
        X_list.append(X_sampled)
        y_list.append(y_sampled)

    X_aug = pd.concat(X_list, axis=0).reset_index(drop=True)
    y_aug = pd.concat(y_list, axis=0).reset_index(drop=True)

    return X_aug, y_aug
