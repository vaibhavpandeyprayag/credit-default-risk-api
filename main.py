from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TARGET_COLUMN, DATA_PATH, TEST_SIZE
from src.pipelines.model_pipelines import get_logistic_regression_pipeline, get_random_forest_pipeline, get_xgboost_pipeline
from src.evaluation import cross_validate_model, evaluate_at_threshold, check_overfitting, save_model_metrics, find_optimal_threshold
from src.data_loader import load_data
from src.data.data_processing import clean_and_prepare_data
import joblib
import json

def main():
    column_names = ["checking_account_status", "duration_months", "credit_history", "purpose", "credit_amount",
                "savings_account", "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
                "residence_since", "property", "age", "other_installment_plans", "housing",
                "existing_credits", "job", "num_liable_people", "telephone", "foreign_worker",
                "risk"]
    df = load_data(DATA_PATH)
    df.columns = column_names
    df = clean_and_prepare_data(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Run CV on full data first — checks stability across splits
    lg_model = get_logistic_regression_pipeline()
    cv_lg = cross_validate_model(lg_model, X, y, "Logistic Regression")
    lg_model.fit(X_train, y_train)
    lg_threshold = find_optimal_threshold(lg_model, X_test, y_test, min_recall=0.7)
    lg_evaluation = evaluate_at_threshold(lg_model, X_test, y_test, lg_threshold)
    print("Logistic Regression Results:")
    print(f"Accuracy: {lg_evaluation['accuracy']}")
    print(f"Precision: {lg_evaluation['precision']}")
    print(f"Recall: {lg_evaluation['recall']}")
    print(f"F1: {lg_evaluation['f1']}")
    print(f"AUC: {lg_evaluation['auc']}")
    print("Report:")
    print(json.dumps(lg_evaluation['report'], indent=4))
    print(f"Confusion Matrix: {lg_evaluation['confusion_matrix']}")
    check_overfitting(lg_model, X_train, y_train, X_test, y_test, "Logistic Regression")
    save_model_metrics("logistic_regression", lg_evaluation, "reports/logistic_regression_evaluation_results.json")
    joblib.dump(lg_model, "models/logistic_regression_model.joblib")
    joblib.dump(lg_threshold, "models/logistic_regression_threshold.joblib")

    print("\n" + "="*50 + "\n")

    rf_model = get_random_forest_pipeline()
    cv_rf = cross_validate_model(rf_model, X, y, "Random Forest")
    rf_model.fit(X_train, y_train)
    rf_threshold = find_optimal_threshold(rf_model, X_test, y_test, min_recall=0.7)
    rf_evaluation = evaluate_at_threshold(rf_model, X_test, y_test, rf_threshold)
    print("Random Forest Results:")
    print(f"Accuracy: {rf_evaluation['accuracy']}")
    print(f"Precision: {rf_evaluation['precision']}")
    print(f"Recall: {rf_evaluation['recall']}")
    print(f"F1: {rf_evaluation['f1']}")
    print(f"AUC: {rf_evaluation['auc']}")
    print("Report:")
    print(json.dumps(rf_evaluation['report'], indent=4))
    print(f"Confusion Matrix: {rf_evaluation['confusion_matrix']}")
    check_overfitting(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
    save_model_metrics("random_forest", rf_evaluation, "reports/random_forest_evaluation_results.json")
    joblib.dump(rf_model, "models/random_forest_model.joblib")
    joblib.dump(rf_threshold, "models/random_forest_threshold.joblib")
    
    print("\n" + "="*50 + "\n")

    xgb_model = get_xgboost_pipeline()
    cv_xgb = cross_validate_model(xgb_model, X, y, "XGBoost")
    xgb_model.fit(X_train, y_train)
    xgb_threshold = find_optimal_threshold(xgb_model, X_test, y_test, min_recall=0.7)
    xgb_evaluation = evaluate_at_threshold(xgb_model, X_test, y_test, xgb_threshold)
    print("XGBoost Results:")
    print(f"Accuracy: {xgb_evaluation['accuracy']}")
    print(f"Precision: {xgb_evaluation['precision']}")
    print(f"Recall: {xgb_evaluation['recall']}")
    print(f"F1: {xgb_evaluation['f1']}")
    print(f"AUC: {xgb_evaluation['auc']}")
    print("Report:")
    print(json.dumps(xgb_evaluation['report'], indent=4))
    print(f"Confusion Matrix: {xgb_evaluation['confusion_matrix']}")   
    check_overfitting(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")
    save_model_metrics("xgboost", xgb_evaluation, "reports/xgboost_evaluation_results.json")
    joblib.dump(xgb_model, "models/xgboost_model.joblib")
    joblib.dump(xgb_threshold, "models/xgboost_threshold.joblib")

if __name__ == "__main__":
    main()