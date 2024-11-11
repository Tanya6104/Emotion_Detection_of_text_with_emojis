import pandas as pd

# 1. Load the Dataset
df = pd.read_csv(r'C:\Users\barcn\Desktop\NLP_PROJECT\emoji_output.csv', encoding='utf-8')
print("Dataset loaded successfully.")

# 2. Extract only the 'Emoji' column
emoji_df = df[['Emoji']]

# 3. Remove duplicates, if any
emoji_df = emoji_df.drop_duplicates()

# 4. Save the extracted emojis to a new CSV file
output_file = r'C:\Users\barcn\Desktop\NLP_PROJECT\emoji_only_output.csv'
emoji_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Emoji data saved to {output_file}.")
