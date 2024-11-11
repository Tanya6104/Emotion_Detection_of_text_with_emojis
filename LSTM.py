import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import os

# Load the dataset
df = pd.read_csv(r'C:\Users\barcn\Desktop\NLP_PROJECT\emoji_output.csv', encoding='utf-8')
print("Dataset loaded successfully.")

# Save the first few rows of the dataset to a file
first_few_rows = df.head()
output_dir = r'C:\Users\barcn\Desktop\NLP_PROJECT'
first_few_rows_output_path = os.path.join(output_dir, 'first_few_rows_output.txt')
with open(first_few_rows_output_path, 'w', encoding='utf-8') as file:
    first_few_rows.to_string(file)
print(f"First few rows of the dataset saved to {first_few_rows_output_path}")

# Data Preprocessing
X = df['Text']  # Feature: Text column
y = df['Label'] # Target: Label column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Define a pipeline with TfidfVectorizer and Naive Bayes classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()), 
    ('nb', MultinomialNB())
])

# Perform 5-fold cross-validation and print the results
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
mean_cv_score = np.mean(cv_scores)
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {mean_cv_score}")

# Save cross-validation results to a file
cv_results_output_path = os.path.join(output_dir, 'cv_results.txt')
with open(cv_results_output_path, 'w', encoding='utf-8') as file:
    file.write(f"Cross-Validation Accuracy Scores: {cv_scores}\n")
    file.write(f"Mean CV Accuracy: {mean_cv_score}\n")
print(f"Cross-validation results saved to {cv_results_output_path}")

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = pipeline.predict(X_test)
classification_report_output = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# Save the classification report to a file
classification_report_output_path = os.path.join(output_dir, 'classification_report.txt')
with open(classification_report_output_path, 'w', encoding='utf-8') as file:
    file.write("Classification Report:\n")
    file.write(classification_report_output)
print(f"Model performance metrics saved to {classification_report_output_path}")

# Save the confusion matrix to a CSV file
confusion_matrix_output_path = os.path.join(output_dir, 'confusion_matrix.csv')
confusion_matrix_df = pd.DataFrame(confusion_mat, index=pipeline.classes_, columns=pipeline.classes_)
confusion_matrix_df.to_csv(confusion_matrix_output_path, encoding='utf-8')
print(f"Confusion matrix saved to {confusion_matrix_output_path}")

# Print a sample of training and test data to verify data processing
print("\nSample of training data:")
print(X_train.sample(5).to_string(index=False))
print("\nSample of test data:")
print(X_test.sample(5).to_string(index=False))
