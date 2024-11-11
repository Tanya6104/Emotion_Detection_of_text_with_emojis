import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np

# 1. Load the Dataset
df = pd.read_csv(r'C:\Users\barcn\Desktop\NLP_PROJECT\emoji_output.csv', encoding='utf-8')

# 2. Save the first few rows to a text file instead of printing
output_file = r'C:\Users\barcn\Desktop\NLP_PROJECT\first_few_rows_output.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("First few rows of the dataset:\n")
    f.write(df.head().to_string())
print(f"First few rows of the dataset saved to {output_file}")

# 3. Prepare features and labels
X = df['Text']  # Use 'Text' column for input features
y = df['Label']  # Use 'Label' column for target labels

# 4. Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# 5. Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Initialize the Logistic Regression classifier
log_reg_classifier = LogisticRegression(random_state=42, max_iter=1000)

# 7. Perform 5-fold cross-validation
cv_scores = cross_val_score(log_reg_classifier, X_train_vec, y_train, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores):.2f}")

# 8. Train the model on the full training data
log_reg_classifier.fit(X_train_vec, y_train)

# 9. Make predictions and evaluate metrics on the test set
y_pred = log_reg_classifier.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# 10. Save results to a text file
metrics_output_file = r'C:\Users\barcn\Desktop\NLP_PROJECT\logistic_regression_performance_metrics.txt'
with open(metrics_output_file, 'w', encoding='utf-8') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")
    f.write(f"F1 Score: {f1:.2f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, zero_division=0))

print(f"Model performance metrics saved to {metrics_output_file}")

# 11. Save confusion matrix to a CSV file
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=np.unique(y_test), columns=np.unique(y_test))
conf_matrix_file = r'C:\Users\barcn\Desktop\NLP_PROJECT\logistic_regression_confusion_matrix.csv'
conf_matrix_df.to_csv(conf_matrix_file)
print(f"Confusion matrix saved to {conf_matrix_file}")

# 12. Print predictions vs actual labels (first 10 samples)
print("\nPredictions vs Actual Labels (first 10 samples):")
for pred, actual in zip(y_pred[:10], y_test[:10]):
    print(f"Predicted: {pred}, Actual: {actual}")

# 13. Inspecting some training and testing data samples
print("\nSample of training data:")
print(X_train.head())
print("\nSample of test data:")
print(X_test.head())
