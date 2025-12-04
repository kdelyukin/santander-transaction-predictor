from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

def train_gnb(X_train, y_train):
    """
    Trains a Gaussian Naive Bayes model.
    """
    print("Training Gaussian Naive Bayes...")
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def evaluate_gnb(model, X_val, y_val):
    """
    Predicts and evaluates the model using AUC score.
    """
    preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, preds)
    print(f"Gaussian NB AUC Score: {score:.5f}")
    return preds, score