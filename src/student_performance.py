import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
# 1. Load dataset
# -----------------------------
DATA_PATH = os.path.join("data", "student_performance.csv")
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n===== STUDENT PERFORMANCE ANALYSIS =====")
print("\nFirst 5 records:")
print(df.head())

# -----------------------------
# 2. Basic data checking
# -----------------------------
print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

# Remove duplicate rows if any
df = df.drop_duplicates()

# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------
print("\nAverage values:")
print(df[[
    "Study_Hours",
    "Attendance_Percent",
    "Previous_Marks",
    "Assignment_Score",
    "Sleep_Hours",
    "Final_Marks"
]].mean())

# Graph 1: Study hours vs final marks
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Study_Hours", y="Final_Marks")
plt.title("Study Hours vs Final Marks")
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "study_hours_vs_marks.png"))
plt.close()

# Graph 2: Correlation heatmap
plt.figure(figsize=(9, 6))
correlation = df.drop(columns=["Student_ID"]).corr()
sns.heatmap(correlation, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Between Student Performance Factors")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# -----------------------------
# 4. Machine Learning
# -----------------------------
features = [
    "Study_Hours",
    "Attendance_Percent",
    "Previous_Marks",
    "Assignment_Score",
    "Sleep_Hours"
]

X = df[features]
y = df["Final_Marks"]

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Model evaluation
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL RESULTS =====")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Graph 3: Actual vs predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
plt.title("Actual vs Predicted Final Marks")
plt.xlabel("Actual Final Marks")
plt.ylabel("Predicted Final Marks")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_actual_vs_predicted.png"))
plt.close()

# -----------------------------
# 6. New student prediction
# -----------------------------
print("\n===== NEW STUDENT PREDICTION =====")
print("Enter student details.")

try:
    study_hours = float(input("Study hours: "))
    attendance = float(input("Attendance percentage: "))
    previous_marks = float(input("Previous marks: "))
    assignment_score = float(input("Assignment score: "))
    sleep_hours = float(input("Sleep hours: "))

    new_student = pd.DataFrame([[
        study_hours,
        attendance,
        previous_marks,
        assignment_score,
        sleep_hours
    ]], columns=features)

    prediction = model.predict(new_student)[0]

    # Keep prediction in a realistic 0-100 range
    prediction = np.clip(prediction, 0, 100)

    print(f"\nPredicted Final Marks: {prediction:.2f}/100")

    if prediction >= 75:
        print("Performance level: Good")
    elif prediction >= 50:
        print("Performance level: Average")
    else:
        print("Performance level: Needs Improvement")

except ValueError:
    print("\nPlease enter numbers only.")

print("\nGraphs saved in the 'outputs' folder.")
print("Project completed successfully.")
