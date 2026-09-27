"""
Loads the train/test split produced by the data-prep job, tunes a
RandomForest classifier, logs the run to MLflow, evaluates the best
model, and saves it so the workflow can commit it to the repository.
"""

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")

def load_data():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])
    return pipeline

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    Xtrain, Xtest, ytrain, ytest = load_data()
    pipeline = build_pipeline(Xtrain)

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }
    mlflow.set_experiment("tourism-wellness-package")
    with mlflow.start_run():
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )
        grid_search.fit(Xtrain, ytrain)
        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)
        ypred = best_model.predict(Xtest)
        metrics = {
            "accuracy": accuracy_score(ytest, ypred),
            "precision": precision_score(ytest, ypred),
            "recall": recall_score(ytest, ypred),
            "f1_score": f1_score(ytest, ypred),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        print("Best hyperparameters:")
        for k, v in grid_search.best_params_.items():
            print(f"  {k}: {v}")
        print("\nTest set performance:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification report:")
        print(classification_report(ytest, ypred))
        joblib.dump(best_model, MODEL_PATH)
        print(f"\nBest model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
