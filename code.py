# -*- coding: utf-8 -*-
"""


@author: manan
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  Load the dataset
df = pd.read_csv("Current_Employee_Names__Salaries__and_Position_Titles (1).csv")

#  Data Cleaning
df.drop_duplicates(inplace=True)  # Remove duplicates
df.columns = df.columns.str.strip()  # Clean column names
df["Full or Part-Time"] = df["Full or Part-Time"].str.strip()
df["Salary or Hourly"] = df["Salary or Hourly"].str.strip()
print(df.info())

#  Split data based on salary type 
salaried = df[df["Salary or Hourly"] == "SALARY"]
hourly = df[df["Salary or Hourly"] == "HOURLY"]

#  Pie Chart - Full-Time vs Part-Time
plt.figure(figsize=(6, 6))
df["Full or Part-Time"].value_counts().plot.pie(autopct="%1.1f%%", colors=["lightblue", "orange"])
plt.title("Full-Time vs Part-Time Employees")
plt.ylabel("")
plt.show()

#  Bar Chart - Top 5 Departments by Employee Count
plt.figure()
df["Department"].value_counts().head(5).plot(kind="bar", color="skyblue")
plt.title("Top 5 Departments by Employee Count")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.xticks(rotation=45)
plt.show()

#  Histogram - Annual Salary Distribution
plt.figure()
sns.histplot(salaried["Annual Salary"], bins=30, kde=True, color="lightgreen")
plt.title("Annual Salary Distribution")
plt.xlabel("Annual Salary")
plt.ylabel("Frequency")
plt.show()

#  Boxplot - Hourly Rate with Outliers
plt.figure()
sns.boxplot(y=hourly["Hourly Rate"], color="tomato")
plt.title("Boxplot of Hourly Rates (with Outliers)")
plt.ylabel("Hourly Rate")
plt.show()

#  Heatmap - Correlation between numeric columns
plt.figure()
sns.heatmap(df[["Annual Salary", "Hourly Rate", "Typical Hours"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


#  Linear Regression Plot - Hourly Rate vs Typical Hours (for hourly employees)
plt.figure()
sns.regplot(x="Typical Hours", y="Hourly Rate", data=hourly, scatter_kws={"color": "lightgreen"}, line_kws={"color": "blue"})
plt.title("Linear Regression: Hourly Rate vs Typical Hours")
plt.xlabel("Typical Hours Worked")
plt.ylabel("Hourly Rate")
plt.show()

#  Function to detect outliers and inliers using IQR
def detect_outliers_inliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    inliers = data[(data[column] >= lower) & (data[column] <= upper)]
    return outliers, inliers

#  Detect outliers and inliers for Annual Salary
salary_outliers, salary_inliers = detect_outliers_inliers(salaried, "Annual Salary")

#  Detect outliers and inliers for Hourly Rate
hourly_outliers, hourly_inliers = detect_outliers_inliers(hourly, "Hourly Rate")

#  Display outlier and inlier counts
print("Annual Salary - Outliers:", len(salary_outliers), "Inliers:", len(salary_inliers))
print("Hourly Rate - Outliers:", len(hourly_outliers), "Inliers:", len(hourly_inliers))

# Summary Statistics - Annual Salary
print("Summary Statistics for Annual Salary:")
print(salaried["Annual Salary"].describe())

#  Summary Statistics - Hourly Rate
print("\nSummary Statistics for Hourly Rate:")
print(hourly["Hourly Rate"].describe())
