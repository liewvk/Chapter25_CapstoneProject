import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)


def load_data(data_file):
    df = pd.read_csv(data_file)
    return df


def clean_data(df):
    df = df.copy()

    numeric_columns = [
        "StudyHours",
        "Attendance",
        "AssignmentScore",
        "PracticeScore",
        "FinalScore"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in numeric_columns:
        average_value = df[column].mean()
        df[column] = df[column].fillna(average_value)

    df = df.drop_duplicates()

    df["Result"] = np.where(df["FinalScore"] >= 50, "Pass", "Fail")

    return df


def create_charts(df, output_folder):
    chart1 = output_folder / "study_hours_vs_final_score.png"
    chart2 = output_folder / "attendance_vs_final_score.png"
    chart3 = output_folder / "final_score_distribution.png"

    plt.figure(figsize=(8, 5))
    plt.scatter(df["StudyHours"], df["FinalScore"])
    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study Hours")
    plt.ylabel("Final Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(chart1)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["Attendance"], df["FinalScore"])
    plt.title("Attendance vs Final Score")
    plt.xlabel("Attendance")
    plt.ylabel("Final Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(chart2)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["FinalScore"], bins=8)
    plt.title("Final Score Distribution")
    plt.xlabel("Final Score")
    plt.ylabel("Number of Students")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(chart3)
    plt.close()


def train_regression_model(df):
    X = df[[
        "StudyHours",
        "Attendance",
        "AssignmentScore",
        "PracticeScore"
    ]]

    y = df["FinalScore"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results = pd.DataFrame({
        "StudyHours": X_test["StudyHours"],
        "Attendance": X_test["Attendance"],
        "AssignmentScore": X_test["AssignmentScore"],
        "PracticeScore": X_test["PracticeScore"],
        "ActualFinalScore": y_test,
        "PredictedFinalScore": predictions.round(2),
        "AbsoluteError": abs(y_test - predictions).round(2)
    })

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

    return model, results, metrics


def train_classification_model(df):
    X = df[[
        "StudyHours",
        "Attendance",
        "AssignmentScore",
        "PracticeScore"
    ]]

    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(y_test, predictions)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["Fail", "Pass"]
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Fail", "Actual Pass"],
        columns=["Predicted Fail", "Predicted Pass"]
    )

    results = pd.DataFrame({
        "StudyHours": X_test["StudyHours"],
        "Attendance": X_test["Attendance"],
        "AssignmentScore": X_test["AssignmentScore"],
        "PracticeScore": X_test["PracticeScore"],
        "ActualResult": y_test,
        "PredictedResult": predictions
    })

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    metrics = {
        "accuracy": accuracy,
        "report": report,
        "matrix": matrix_df
    }

    return model, results, metrics, feature_importance


def predict_new_students(regression_model, classification_model):
    new_students = pd.DataFrame({
        "StudyHours": [6, 3, 9],
        "Attendance": [78, 58, 94],
        "AssignmentScore": [70, 45, 90],
        "PracticeScore": [72, 42, 92]
    })

    predicted_scores = regression_model.predict(new_students)
    predicted_results = classification_model.predict(new_students)

    probabilities = classification_model.predict_proba(new_students)

    prediction_results = new_students.copy()
    prediction_results["PredictedFinalScore"] = predicted_scores.round(2)
    prediction_results["PredictedResult"] = predicted_results

    pass_probabilities = []

    for class_names, probability_values in [
        (classification_model.classes_, row)
        for row in probabilities
    ]:
        probability_dictionary = dict(zip(class_names, probability_values))
        pass_probabilities.append(
            probability_dictionary.get("Pass", 0) * 100
        )

    prediction_results["PassProbability"] = [
        round(value, 2)
        for value in pass_probabilities
    ]

    return prediction_results


def save_feature_importance_chart(feature_importance, output_folder):
    chart_file = output_folder / "feature_importance.png"

    plt.figure(figsize=(8, 5))
    plt.bar(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )
    plt.title("Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(chart_file)
    plt.close()


def save_report(
    output_folder,
    df,
    regression_metrics,
    classification_metrics,
    feature_importance
):
    report_file = output_folder / "capstone_report.txt"

    report_text = f"""
Student Performance AI Assistant Report
======================================

Dataset Summary
---------------
Total records: {len(df)}
Columns: {list(df.columns)}

Result Counts
-------------
{df["Result"].value_counts()}

Regression Model Evaluation
---------------------------
Mean Absolute Error: {regression_metrics["mae"]:.2f}
Mean Squared Error: {regression_metrics["mse"]:.2f}
Root Mean Squared Error: {regression_metrics["rmse"]:.2f}
R-squared Score: {regression_metrics["r2"]:.2f}

Classification Model Evaluation
-------------------------------
Accuracy: {classification_metrics["accuracy"]:.2f}
Accuracy Percentage: {classification_metrics["accuracy"] * 100:.2f}%

Confusion Matrix
----------------
{classification_metrics["matrix"]}

Classification Report
---------------------
{classification_metrics["report"]}

Feature Importance
------------------
{feature_importance}
"""

    with open(report_file, "w") as file:
        file.write(report_text)


def main():
    data_file = Path("data") / "student_performance.csv"
    output_folder = Path("outputs")

    output_folder.mkdir(exist_ok=True)

    df = load_data(data_file)

    print("Original Dataset")
    print("----------------")
    print(df.head())

    clean_df = clean_data(df)

    print()
    print("Cleaned Dataset")
    print("---------------")
    print(clean_df.head())

    print()
    print("Result Counts")
    print("-------------")
    print(clean_df["Result"].value_counts())

    clean_data_file = output_folder / "cleaned_student_performance.csv"
    clean_df.to_csv(clean_data_file, index=False)

    create_charts(clean_df, output_folder)

    regression_model, regression_results, regression_metrics = train_regression_model(
        clean_df
    )

    classification_model, classification_results, classification_metrics, feature_importance = train_classification_model(
        clean_df
    )

    new_student_predictions = predict_new_students(
        regression_model,
        classification_model
    )

    save_feature_importance_chart(feature_importance, output_folder)

    regression_results.to_csv(
        output_folder / "regression_predictions.csv",
        index=False
    )

    classification_results.to_csv(
        output_folder / "classification_predictions.csv",
        index=False
    )

    feature_importance.to_csv(
        output_folder / "feature_importance.csv",
        index=False
    )

    new_student_predictions.to_csv(
        output_folder / "new_student_predictions.csv",
        index=False
    )

    save_report(
        output_folder,
        clean_df,
        regression_metrics,
        classification_metrics,
        feature_importance
    )

    print()
    print("Regression Results")
    print("------------------")
    print(regression_results)

    print()
    print("Regression Evaluation")
    print("---------------------")
    print(f"Mean Absolute Error: {regression_metrics['mae']:.2f}")
    print(f"Root Mean Squared Error: {regression_metrics['rmse']:.2f}")
    print(f"R-squared Score: {regression_metrics['r2']:.2f}")

    print()
    print("Classification Results")
    print("----------------------")
    print(classification_results)

    print()
    print("Classification Evaluation")
    print("-------------------------")
    print(f"Accuracy: {classification_metrics['accuracy']:.2f}")
    print(f"Accuracy Percentage: {classification_metrics['accuracy'] * 100:.2f}%")

    print()
    print("Confusion Matrix")
    print("----------------")
    print(classification_metrics["matrix"])

    print()
    print("Feature Importance")
    print("------------------")
    print(feature_importance)

    print()
    print("New Student Predictions")
    print("-----------------------")
    print(new_student_predictions)

    print()
    print("Capstone project completed.")
    print("All output files have been saved in the outputs folder.")


main()
