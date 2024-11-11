import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import sys

# Load the Dataset
df = pd.read_csv(r'C:\Users\barcn\Desktop\NLP_PROJECT\emoji_output.csv', encoding='utf-8')
print("Dataset loaded successfully.")
print(df.head())  # Display first few rows to confirm data is loaded

# Data Preprocessing
X = df['Text']       # Text column
y = df['Label']      # Label column

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline with TfidfVectorizer and Logistic Regression Classifier
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # Convert text to TF-IDF features
    ('classifier', LogisticRegression(max_iter=200, random_state=42))  # Apply Logistic Regression classifier
])

# Set up parameter grid for GridSearchCV
param_grid = {
    'classifier__C': [0.1, 1, 10],              # Inverse of regularization strength
    'classifier__penalty': ['l2'],               # L2 penalty
    'classifier__solver': ['liblinear']          # Solver for small datasets
}

# Perform grid search to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Display best parameters found from grid search
print("Best parameters found:", grid_search.best_params_)

# Predict using the best model from grid search
y_pred = grid_search.predict(X_test)

# Generate the classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Generate and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Logistic Regression):\n", cm)  # To double-check values in the console
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Logistic Regression)")

# Create output directory if it does not exist
output_dir = "confusion_matrices_output"
os.makedirs(output_dir, exist_ok=True)

# Save the plot as an image with the name of the algorithm
output_path = os.path.join(output_dir, "Logistic_Regression_Confusion_Matrix.png")
plt.savefig(output_path)
print(f"Confusion matrix image saved as: {output_path}")

# Show the plot without blocking and close it
plt.show(block=False)
plt.close()

# Exit the script
sys.exit()
