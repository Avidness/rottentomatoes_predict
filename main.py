from data_loader import load_and_preprocess_data
from models import xgboost_model, random_forest_model, linear_regression_model, neural_network_model
from utils import evaluate_model, print_results

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train and evaluate models
    models = [
        ("XGBoost", xgboost_model),
        ("Random Forest", random_forest_model),
        ("Linear Regression", linear_regression_model),
        ("Neural Network", neural_network_model)
    ]

    for model_name, model_module in models:
        y_pred = model_module.train_and_predict(X_train, y_train, X_test)
        rmse, r2 = evaluate_model(y_test, y_pred)
        print_results(model_name, rmse, r2)

if __name__ == "__main__":
    main()

