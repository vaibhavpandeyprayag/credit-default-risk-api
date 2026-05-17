import os
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_validate

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc': roc_auc_score(y, y_prob),
        'report': classification_report(y, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y, y_pred)
    }
    
    return metrics

def find_optimal_threshold(model, X_val, y_val, min_recall=0.70):
    """
    Finds the highest threshold where recall >= min_recall.
    More intuitive for credit risk — set a recall floor, maximize precision above it.
    """
    probs = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.10, 0.90, 0.01)
    best_threshold = 0.5  # fallback
    best_precision = 0.0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        r = recall_score(y_val, preds)
        p = precision_score(y_val, preds)
        if r >= min_recall and p > best_precision:
            best_precision = p
            best_threshold = t

    probs_at_best = (probs >= best_threshold).astype(int)
    final_recall = recall_score(y_val, probs_at_best)
    final_precision = precision_score(y_val, probs_at_best)

    print(f"\nOptimal Threshold: {best_threshold:.4f}")
    print(f"At this threshold → Precision: {final_precision:.4f} | Recall: {final_recall:.4f}")

    return best_threshold
    
def evaluate_at_threshold(model, X_test, y_test, threshold):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'auc': roc_auc_score(y_test, probs),
        'report': classification_report(y_test, preds, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, preds)
    }

    return metrics


def check_overfitting(model, X_train, y_train, X_test, y_test, name):
    
    # Train performance
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    train_acc = roc_auc_score(y_train, y_train_prob)

    # Test performance
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_acc = roc_auc_score(y_test, y_test_prob)

    print(f"\n{name}")
    print("Train AUC:", train_acc)
    print("Test AUC:", test_acc)
    print("Gap:", train_acc - test_acc)


def cross_validate_model(pipeline, X, y, name, cv=5):
    """
    Runs stratified k-fold CV on the full dataset (before train/test split).
    Uses pipeline directly — no data leakage since preprocessing is inside pipeline.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, return_train_score=True)

    print(f"\n--- {name} | {cv}-Fold Stratified CV ---")
    for metric in scoring:
        test_scores = results[f'test_{metric}']
        train_scores = results[f'train_{metric}']
        print(f"{metric:12s}: test={test_scores.mean():.4f} ± {test_scores.std():.4f} | "
              f"train={train_scores.mean():.4f} | gap={train_scores.mean() - test_scores.mean():.4f}")

    return {
        f'cv_{metric}': results[f'test_{metric}'].mean()
        for metric in scoring
    }



def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

def save_model_metrics(model_name, metrics, path):
    os.makedirs("reports", exist_ok=True)
    data = {}
    # # Load existing
    # if os.path.exists(path):
    #     with open(path, "r") as f:
    #         data = json.load(f)
    # else:
    #     data = {}

    # Convert all values

    cleaned_metrics = {k: convert_numpy(v) for k, v in metrics.items()}
    # Add current model results
    data[model_name] = cleaned_metrics

    # Save back
    with open(path, "w") as f:
        json.dump(data, f, indent=4)