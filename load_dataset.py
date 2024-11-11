import pandas as pd
import re

# 1. Load the Dataset
df = pd.read_csv(r'C:\Users\barcn\Desktop\NLP_PROJECT\dataset.csv', encoding='utf-8')
print("Dataset loaded successfully.")

# 2. Text Cleaning
# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s\U0001F600-\U0001F64F]', '', text)  # Remove punctuation, keep emojis
    return text

# Apply the cleaning function to the 'Text' column
df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Display cleaned text for verification
print("\nCleaned Text (First Few Rows):")
try:
    print(df[['Text', 'Cleaned_Text']].head().to_string(index=False))
except Exception as e:
    print("Error displaying the DataFrame:", e)

# Save the DataFrame with cleaned text and emojis to a new CSV file
output_file = r'C:\Users\barcn\Desktop\NLP_PROJECT\emoji_output.csv'
df[['Text', 'Emoji', 'Label', 'Cleaned_Text']].to_csv(output_file, index=False, encoding='utf-8')
print(f"\nEmoji data saved to {output_file}.")
