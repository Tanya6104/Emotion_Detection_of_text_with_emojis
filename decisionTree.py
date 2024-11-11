import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np

# 1. Load the Dataset
df = pd.read_csv(r'C:\Users\barcn\Desktop\NLP_PROJECT\emoji_output.csv', encoding='utf-8')

# 2. Save the first few rows to a text file instead of printing
output_file = r'C:\Users\barcn\Desktop\NLP_PROJECT\first_few_rows_output.txt'

# Writing the first few rows of the dataset to the text file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("First few rows of the dataset:\n")
    f.write(df.head().to_string())  # Convert the DataFrame to string and write to file

print(f"First few rows of the dataset saved to {output_file}")

# 3. Prepare features and labels
X = df['Text']  # Use 'Text' column for input features
y = df['Label']  # Use 'Label' column for target labels

# 4. Vectorize the text data using CountVectorizer (converts text to numeric form)
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)  # Fit and transform the entire dataset

# 5. Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# 6. Perform 5-fold cross-validation and calculate evaluation metrics
accuracy_scores = cross_val_score(dt_classifier, X_vec, y, cv=5, scoring='accuracy')
precision_scores = cross_val_score(dt_classifier, X_vec, y, cv=5, scoring='precision_weighted')
recall_scores = cross_val_score(dt_classifier, X_vec, y, cv=5, scoring='recall_weighted')
f1_scores = cross_val_score(dt_classifier, X_vec, y, cv=5, scoring='f1_weighted')

# 7. Calculate the mean of the evaluation metrics
mean_accuracy = np.mean(accuracy_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)

# 8. Save results to a text file
metrics_output_file = r'C:\Users\barcn\Desktop\NLP_PROJECT\decision_tree_performance_metrics.txt'
with open(metrics_output_file, 'w', encoding='utf-8') as f:
    f.write(f"Mean Accuracy: {mean_accuracy:.2f}\n")
    f.write(f"Mean Precision: {mean_precision:.2f}\n")
    f.write(f"Mean Recall: {mean_recall:.2f}\n")
    f.write(f"Mean F1 Score: {mean_f1:.2f}\n")

print(f"Model performance metrics saved to {metrics_output_file}")

# 9. Train the model on the entire dataset for later predictions (optional)
dt_classifier.fit(X_vec, y)

# 10. Make predictions using the test data (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_test_vec = vectorizer.transform(X_test)  # Transform test data

# 11. Make predictions using the test data
y_pred = dt_classifier.predict(X_test_vec)

# 12. Save confusion matrix to a CSV file
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=np.unique(y_test), columns=np.unique(y_test))
conf_matrix_file = r'C:\Users\barcn\Desktop\NLP_PROJECT\decision_tree_confusion_matrix.csv'
conf_matrix_df.to_csv(conf_matrix_file)
print(f"Confusion matrix saved to {conf_matrix_file}")

# 13. Limit predictions output to the first few entries
print("\nPredictions vs Actual Labels (first 10 samples):")
for pred, actual in zip(y_pred[:10], y_test[:10]):
    print(f"Predicted: {pred}, Actual: {actual}")

# 14. Inspecting some training and testing data samples
print("\nSample of training data:")
print(X_train.head())
print("\nSample of test data:")
print(X_test.head())
