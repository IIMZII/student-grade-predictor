# student-grade-predictor
# A simple machine learning project that predicts if a student may pass or fail

from sklearn.tree import DecisionTreeClassifier
import pandas as pd


# Sample student data
# columns: study_hours, attendance_percent, previous_score, result
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 9, 10, 3, 6, 7],
    "attendance_percent": [40, 50, 55, 60, 70, 75, 80, 90, 45, 95, 98, 58, 78, 85],
    "previous_score": [30, 40, 45, 50, 60, 65, 70, 85, 35, 90, 95, 48, 68, 75],
    "result": [
        "Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Pass", "Fail", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

# Features and target
X = df[["study_hours", "attendance_percent", "previous_score"]]
y = df["result"]

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X, y)


def predict_student_result():
    print("===================================")
    print("      Student Grade Predictor")
    print("===================================")

    try:
        study_hours = float(input("Enter study hours per day: "))
        attendance = float(input("Enter attendance percentage: "))
        previous_score = float(input("Enter previous score: "))

        student_data = [[study_hours, attendance, previous_score]]
        prediction = model.predict(student_data)

        print("\nPrediction Result:")
        print("------------------")
        print(f"The student is predicted to: {prediction[0]}")

    except ValueError:
        print("Invalid input. Please enter numbers only.")


if __name__ == "__main__":
    predict_student_result()
