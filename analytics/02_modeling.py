import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")


# -------------------------------------------------
# 1. PATH CONFIGURATION
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "titanic_cleaned.csv")


# -------------------------------------------------
# 2. LOAD CLEANED DATASET
# -------------------------------------------------

def load_data():

    print("\nSTEP 1: LOADING CLEANED TITANIC DATASET")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Cleaned Titanic dataset not found. "
            "Run analytics/01_eda.py first."
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nDataset preview:")
    print(df.head())

    return df


# -------------------------------------------------
# 3. CREATE PREPROCESSING PIPELINE
# -------------------------------------------------

def create_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


# -------------------------------------------------
# 4. CLASSIFICATION DATA PREPARATION
# -------------------------------------------------

def prepare_classification_data(df):

    print("\nSTEP 2: CLASSIFICATION DATA PREPARATION")

    if "survived" not in df.columns:
        raise ValueError("The survived target column is missing.")

    # Remove target leakage columns.
    # 'alive' directly describes survival and must not be used.
    # 'class' duplicates the pclass information.
    X = df.drop(
        columns=["survived", "alive", "class"],
        errors="ignore"
    )

    y = df["survived"].astype(int)

    # Split BEFORE fitting any preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True).round(3))

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True).round(3))

    preprocessor = create_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor


# -------------------------------------------------
# 5. CLASSIFICATION EVALUATION
# -------------------------------------------------

def evaluate_classifier(
    name,
    model,
    X_test,
    y_test,
    results,
    output_prefix
):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    })

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"]
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{output_prefix}_confusion_matrix.png"
        )
    )

    plt.close()

    return probabilities


# -------------------------------------------------
# 6. TRAIN THREE CLASSIFICATION MODELS
# -------------------------------------------------

def train_classification_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    print("\nSTEP 3: TRAINING CLASSIFICATION MODELS")

    results = []
    fitted_models = {}

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    }

    for name, estimator in models.items():

        print("\nTraining:", name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator)
            ]
        )

        pipeline.fit(X_train, y_train)

        probabilities = evaluate_classifier(
            name,
            pipeline,
            X_test,
            y_test,
            results,
            name.lower().replace(" ", "_")
        )

        fitted_models[name] = pipeline

        # Save ROC curve values for comparison.
        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc_score(y_test, probabilities):.3f})"
        )

    # Combined ROC curve
    plt.figure(figsize=(8, 6))

    for name, pipeline in fitted_models.items():

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        auc = roc_auc_score(y_test, probabilities)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "roc_comparison.png")
    )

    plt.close()

    return results, fitted_models


# -------------------------------------------------
# 7. DECISION TREE VISUALIZATION
# -------------------------------------------------

def visualize_decision_tree(tree_pipeline):

    print("\nSTEP 4: DECISION TREE VISUALIZATION")

    fitted_preprocessor = tree_pipeline.named_steps[
        "preprocessor"
    ]

    fitted_tree = tree_pipeline.named_steps["model"]

    feature_names = fitted_preprocessor.get_feature_names_out()

    plt.figure(figsize=(24, 12))

    plot_tree(
        fitted_tree,
        feature_names=feature_names,
        class_names=["Not Survived", "Survived"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8
    )

    plt.title("Titanic Decision Tree (First Three Levels)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "decision_tree.png"),
        dpi=200
    )

    plt.close()

    print("Decision tree diagram saved.")


# -------------------------------------------------
# 8. CLASS IMBALANCE COMPARISON
# -------------------------------------------------

def class_imbalance_analysis(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\nSTEP 5: CLASS IMBALANCE ANALYSIS")

    imbalance_results = []

    preprocessor = create_preprocessor(X_train)

    # A. Baseline Logistic Regression
    baseline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    )

    baseline.fit(X_train, y_train)

    evaluate_classifier(
        "Imbalance Baseline",
        baseline,
        X_test,
        y_test,
        imbalance_results,
        "imbalance_baseline"
    )

    # B. Class-weight balanced Random Forest
    balanced = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    balanced.fit(X_train, y_train)

    evaluate_classifier(
        "Random Forest - Class Weight Balanced",
        balanced,
        X_test,
        y_test,
        imbalance_results,
        "class_weight_balanced"
    )

    # C. SMOTE applied ONLY to training data.
    smote_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "smote",
                SMOTE(random_state=42)
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    )

    smote_pipeline.fit(X_train, y_train)

    evaluate_classifier(
        "Logistic Regression - SMOTE",
        smote_pipeline,
        X_test,
        y_test,
        imbalance_results,
        "smote"
    )

    results_df = pd.DataFrame(imbalance_results)

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "class_imbalance_comparison.csv"
        ),
        index=False
    )

    print("\nClass imbalance comparison:")
    print(results_df.round(4).to_string(index=False))

    return results_df


# -------------------------------------------------
# 9. RANDOM FOREST GRID SEARCH
# -------------------------------------------------

def optimize_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    print("\nSTEP 6: RANDOM FOREST GRID SEARCH")

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    bootstrap=True,
                    oob_score=True,
                    random_state=42
                )
            )
        ]
    )

    parameter_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10],
        "model__max_features": ["sqrt", "log2"]
    }

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=parameter_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nBest cross-validation F1 score:")
    print(round(grid_search.best_score_, 4))

    print("\nRandom Forest OOB score:")
    print(
        round(
            best_model.named_steps["model"].oob_score_,
            4
        )
    )

    grid_results = pd.DataFrame(
        grid_search.cv_results_
    )

    grid_results.to_csv(
        os.path.join(OUTPUT_DIR, "grid_search_results.csv"),
        index=False
    )

    # Evaluate tuned model on held-out test data.
    tuned_results = []

    evaluate_classifier(
        "Tuned Random Forest",
        best_model,
        X_test,
        y_test,
        tuned_results,
        "tuned_random_forest"
    )

    return best_model


# -------------------------------------------------
# 10. SAVE AND RELOAD COMPLETE MODEL PIPELINE
# -------------------------------------------------

def save_and_test_model(best_model, X_test):

    print("\nSTEP 7: SAVING AND RELOADING MODEL")

    model_path = os.path.join(
        MODEL_DIR,
        "titanic_survival_pipeline.joblib"
    )

    # Save complete preprocessing + model pipeline.
    joblib.dump(best_model, model_path)

    print("Model saved:", model_path)

    # Reload the saved pipeline.
    loaded_model = joblib.load(model_path)

    # Test prediction using a raw, unprocessed input row.
    raw_input = X_test.iloc[[0]].copy()

    prediction = loaded_model.predict(raw_input)

    probability = loaded_model.predict_proba(raw_input)

    print("\nRaw input test:")
    print(raw_input)

    print("\nReloaded model prediction:", int(prediction[0]))

    print(
        "Probability of survival:",
        round(float(probability[0][1]), 4)
    )

    print("Saved model successfully reloaded and tested.")


# -------------------------------------------------
# 11. FARE REGRESSION DATA PREPARATION
# -------------------------------------------------

def prepare_regression_data(df):

    print("\nSTEP 8: FARE REGRESSION PREPARATION")

    if "fare" not in df.columns:
        raise ValueError("Fare target column is missing.")

    # Predict fare using passenger information.
    # Do not use fare itself or survival-related information.
    X = df.drop(
        columns=["fare", "survived", "alive", "class"],
        errors="ignore"
    )

    y = df["fare"].astype(float)

    # Separate training and test data before preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    preprocessor = create_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor


# -------------------------------------------------
# 12. FARE REGRESSION MODEL
# -------------------------------------------------

def train_fare_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    print("\nSTEP 9: TRAINING FARE REGRESSION MODEL")

    regression_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    regression_pipeline.fit(X_train, y_train)

    predictions = regression_pipeline.predict(X_test)

    # Regression metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    # Number of predictors after preprocessing.
    p = regression_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out().shape[0]

    n = len(y_test)

    if n > p + 1:
        adjusted_r2 = 1 - (
            (1 - r2) * (n - 1) / (n - p - 1)
        )
    else:
        adjusted_r2 = np.nan

    print("\nFARE REGRESSION RESULTS")

    print("Mean Absolute Error (MAE):", round(mae, 4))
    print("Root Mean Squared Error (RMSE):", round(rmse, 4))
    print("R-squared:", round(r2, 4))
    print("Adjusted R-squared:", round(adjusted_r2, 4)
          if not np.isnan(adjusted_r2) else "Not defined")

    # Residuals
    residuals = y_test - predictions

    # Residual plot
    plt.figure(figsize=(8, 5))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.6
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.title("Fare Prediction Residual Plot")
    plt.xlabel("Predicted Fare")
    plt.ylabel("Residual")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "fare_residual_plot.png")
    )

    plt.close()

    print("\nResidual analysis:")

    print(
        "If residual spread increases with predicted fare, "
        "this may indicate heteroscedasticity."
    )

    print(
        "A random, evenly spread residual pattern around zero "
        "is generally desirable."
    )

    # Save regression metrics.
    regression_metrics = pd.DataFrame([{
        "Model": "Random Forest Regressor",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2
    }])

    regression_metrics.to_csv(
        os.path.join(OUTPUT_DIR, "regression_metrics.csv"),
        index=False
    )

    # Save regression pipeline.
    regression_path = os.path.join(
        MODEL_DIR,
        "titanic_fare_regression_pipeline.joblib"
    )

    joblib.dump(
        regression_pipeline,
        regression_path
    )

    print("\nRegression model saved:", regression_path)


# -------------------------------------------------
# 13. FINAL MODEL RECOMMENDATION
# -------------------------------------------------

def final_recommendation(classification_results):

    print("\nSTEP 10: FINAL MODEL RECOMMENDATION")

    results_df = pd.DataFrame(classification_results)

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "classification_metrics.csv"),
        index=False
    )

    print("\nClassification model comparison:")
    print(results_df.round(4).to_string(index=False))

    best_model = results_df.sort_values(
        by="F1_Score",
        ascending=False
    ).iloc[0]

    print("\nBest model based on F1-score:")
    print(best_model["Model"])

    print(
        "\nRecommendation: Consider the F1-score, recall, "
        "ROC-AUC and confusion matrix together before selecting "
        "the final model. The best model should also be suitable "
        "for the cost of false positives and false negatives."
    )


# -------------------------------------------------
# 14. MAIN EXECUTION
# -------------------------------------------------

def main():

    print("=" * 65)
    print("ZEPTO DATA & AI PLATFORM - MACHINE LEARNING")
    print("=" * 65)

    df = load_data()

    # Classification
    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_classification_data(df)
    )

    classification_results, fitted_models = (
        train_classification_models(
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor
        )
    )

    visualize_decision_tree(
        fitted_models["Decision Tree"]
    )

    class_imbalance_analysis(
        X_train,
        X_test,
        y_train,
        y_test
    )

    best_model = optimize_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    save_and_test_model(
        best_model,
        X_test
    )

    final_recommendation(classification_results)

    # Regression
    reg_X_train, reg_X_test, reg_y_train, reg_y_test, reg_preprocessor = (
        prepare_regression_data(df)
    )

    train_fare_regression(
        reg_X_train,
        reg_X_test,
        reg_y_train,
        reg_y_test,
        reg_preprocessor
    )

    print("\n" + "=" * 65)
    print("MACHINE LEARNING ANALYSIS COMPLETED")
    print("Classification and regression results saved.")
    print("=" * 65)


if __name__ == "__main__":
    main()
