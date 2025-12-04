from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def train_logreg(X_train, y_train):
    """
    Trains a Logistic Regression model. 
    """
    print("Training Logistic Regression...")
    
    # Standardize features (Mean=0, Std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # C=1.0 is the inverse regularization strength
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def evaluate_logreg(model, scaler, X_val, y_val):
    """
    Scales validation data using the training scaler, then predicts.
    """
    X_val_scaled = scaler.transform(X_val)
    preds = model.predict_proba(X_val_scaled)[:, 1]
    score = roc_auc_score(y_val, preds)
    print(f"Logistic Regression AUC Score: {score:.5f}")
    return preds, score