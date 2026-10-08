# Student Performance Analysis & Prediction

A beginner-friendly Data Science and Machine Learning project built with Python.

## Project goal
Analyze student performance and predict a student's expected final marks using:
- Study hours
- Attendance
- Previous marks
- Assignment score
- Sleep hours

## Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Folder structure

```text
Student_Performance_Analysis_Prediction/
│
├── data/
│   └── student_performance.csv
│
├── src/
│   └── student_performance.py
│
├── outputs/
│   ├── correlation_heatmap.png
│   ├── study_hours_vs_marks.png
│   └── model_actual_vs_predicted.png
│
├── requirements.txt
└── README.md
```

## How to run on Windows

### 1. Open the project in VS Code
Open the `Student_Performance_Analysis_Prediction` folder.

### 2. Open Terminal

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install libraries

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python src/student_performance.py
```

The program will:
1. Load the CSV dataset
2. Check and clean the data
3. Display basic statistics
4. Create visualizations
5. Train a Linear Regression model
6. Show MAE and R2 score
7. Predict final marks for a sample student
8. Save graphs inside the `outputs` folder

## Example prediction

The program asks for:
- Study hours
- Attendance
- Previous marks
- Assignment score
- Sleep hours

Then it predicts the expected final marks.

## Machine Learning model

This project uses **Linear Regression** because it is simple and easy for a fresher to understand and explain in an interview.

## Important learning points

You should be able to explain:
- What is a dataset?
- What is data cleaning?
- What is EDA?
- What is correlation?
- What are features and target?
- What is train/test split?
- What is Linear Regression?
- What are MAE and R2 score?

## Resume description

**Student Performance Analysis & Prediction**  
*Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn*

- Analyzed 30 student records using Python and Pandas to identify factors affecting academic performance.
- Performed data cleaning and exploratory data analysis using statistical summaries and visualizations.
- Built a Linear Regression model using Scikit-learn to predict final student marks.
- Evaluated model performance using MAE and R² metrics and generated charts for performance analysis.

## Note
This is a learning/project dataset created for practice. It is not real student data.
