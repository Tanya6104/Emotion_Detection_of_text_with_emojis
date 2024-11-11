import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset (replace with your actual dataset path)
df = pd.read_csv(r'C:\Users\barcn\Desktop\NLP_PROJECT\emoji_output.csv', encoding='utf-8')

# 2. Extract the text and labels
X_text = df['Text']  # Use the 'Text' column as features
y = df['Label']      # Use the 'Label' column as target labels

# 3. Label encoding to convert categorical labels to numeric form
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Split the dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_text, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 5. Initialize the TfidfVectorizer to convert text data into TF-IDF features
vectorizer = TfidfVectorizer()

# 6. Fit the vectorizer on training data and transform both training and test data
X_train_vec = vectorizer.fit_transform(X_train)  # Fit on train data
X_test_vec = vectorizer.transform(X_test)  # Transform test data (use transform, not fit_transform)

# 7. Output the shape of the transformed data for confirmation
print(f"Training data shape: {X_train_vec.shape}")
print(f"Test data shape: {X_test_vec.shape}")

# 8. Optional: Check the first few TF-IDF feature names (words)
feature_names = vectorizer.get_feature_names_out()
print(f"First 10 TF-IDF feature names: {feature_names[:10]}")

