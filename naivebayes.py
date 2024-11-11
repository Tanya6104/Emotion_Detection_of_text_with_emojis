import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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

# 4. Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Print the sizes of the training and testing sets
print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# 6. Vectorize the text data using CountVectorizer (converts text to numeric form)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)  # Fit and transform training data
X_test_vec = vectorizer.transform(X_test)  # Transform test data

# 7. Initialize the Naive Bayes classifier
nb_classifier = MultinomialNB()

# 8. Perform 5-fold cross-validation
cv_scores = cross_val_score(nb_classifier, X_train_vec, y_train, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")

# 9. Train the model using training data
nb_classifier.fit(X_train_vec, y_train)

# 10. Make predictions using the test data
y_pred = nb_classifier.predict(X_test_vec)

# 11. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# 12. Save results to a text file
metrics_output_file = r'C:\Users\barcn\Desktop\NLP_PROJECT\model_performance_metrics.txt'
with open(metrics_output_file, 'w', encoding='utf-8') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")
    f.write(f"F1 Score: {f1:.2f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, zero_division=0))

print(f"Model performance metrics saved to {metrics_output_file}")

# 13. Save confusion matrix to a CSV file for easy viewing
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=np.unique(y_test), columns=np.unique(y_test))
conf_matrix_file = r'C:\Users\barcn\Desktop\NLP_PROJECT\confusion_matrix.csv'
conf_matrix_df.to_csv(conf_matrix_file)
print(f"Confusion matrix saved to {conf_matrix_file}")

# 14. Inspect some samples from training and testing data
print("\nSample of training data:")
print(X_train.head())
print("\nSample of test data:")
print(X_test.head())
