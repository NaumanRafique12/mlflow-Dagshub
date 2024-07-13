import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
# Load the Iris dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import  seaborn as sns
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
max_depth = 1
mlflow.set_experiment("dt")

# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
import dagshub
dagshub.init(repo_owner='NaumanRafique12', repo_name='mlflow-Dagshub', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/NaumanRafique12/mlflow-Dagshub.mlflow")
with mlflow.start_run(run_name="final_with_model_tag_code"): # Train the Random Forest classifier
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("max_depth",max_depth)
    mlflow.log_metric("accuracy",accuracy)

    print(f'Accuracy: {accuracy:.2f}')

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("cf.png")
    mlflow.log_artifact("cf.png")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(clf,"decisiontree")
    mlflow.set_tag("author","noman")
    mlflow.set_tag("model","dt")
    plt.show()
