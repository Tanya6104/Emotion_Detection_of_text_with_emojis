import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load your dataset
df = pd.read_csv(r'C:\Users\barcn\Desktop\NLP_PROJECT\emoji_output.csv', encoding='utf-8')

# Step 2: Feature Extraction - Vectorize the text data using TfidfVectorizer
# Convert the 'Text' column into a numerical format
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Text'])  # Assuming 'Text' column contains the text data
y = df['Label']  # Assuming 'Label' column contains the target labels

# Step 3: Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print the sizes of the training and test sets
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Print a few samples from the training data
print("\nSample training data:")
print(X_train[:5])
print("\nSample training labels:")
print(y_train[:5])
