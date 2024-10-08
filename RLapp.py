import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
st.title("Heart Disease Prediction Dashboard")

df = pd.read_csv('heart.csv')
# Convert the DataFrame to a JSON file
df.to_json('heart.json', orient='records', lines=False)

print("Data has been exported to heart.json")

st.write("### Dataset")
st.dataframe(df.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:-1], df.iloc[:, -1], test_size=0.2, random_state=2)

# Models
clf1 = LogisticRegression(max_iter=1000)
clf2 = DecisionTreeClassifier()

# Fit models
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

# Predictions
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

# Model Evaluation
st.write("## Model Accuracy")
st.write(f"**Accuracy of Logistic Regression**: {accuracy_score(y_test, y_pred1):.2f}")
st.write(f"**Accuracy of Decision Trees**: {accuracy_score(y_test, y_pred2):.2f}")

# Confusion Matrix
st.write("### Logistic Regression Confusion Matrix")
cm1 = confusion_matrix(y_test, y_pred1)
sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues")
st.pyplot()

st.write("### Decision Tree Confusion Matrix")
cm2 = confusion_matrix(y_test, y_pred2)
sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues")
st.pyplot()

# Model Metrics
st.write("## Logistic Regression Metrics")
st.write(f"Precision: {precision_score(y_test, y_pred1):.2f}")
st.write(f"Recall: {recall_score(y_test, y_pred1):.2f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred1):.2f}")

st.write("## Decision Tree Metrics")
st.write(f"Precision: {precision_score(y_test, y_pred2):.2f}")
st.write(f"Recall: {recall_score(y_test, y_pred2):.2f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred2):.2f}")

# Sample predictions
st.write("### Sample Predictions")
result = pd.DataFrame()
result['Actual Label'] = y_test.values
result['Logistic Regression Prediction'] = y_pred1
result['Decision Tree Prediction'] = y_pred2
st.write(result.sample(10))
