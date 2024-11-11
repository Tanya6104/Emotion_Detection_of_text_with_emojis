import csv

# Define the data with more entries for better algorithm performance
data = [
    {"Text": "I love this!", "Emoji": "😊", "Label": "Positive"},
    {"Text": "This is so frustrating!", "Emoji": "😡", "Label": "Negative"},
    {"Text": "It's okay, not great.", "Emoji": "😐", "Label": "Neutral"},
    {"Text": "What a wonderful experience!", "Emoji": "😍", "Label": "Positive"},
    {"Text": "Feeling down about the situation.", "Emoji": "😢", "Label": "Negative"},
    {"Text": "So much to do, but I'm excited!", "Emoji": "😄", "Label": "Positive"},
    {"Text": "This is boring...", "Emoji": "😴", "Label": "Negative"},
    {"Text": "I feel indifferent about this.", "Emoji": "😶", "Label": "Neutral"},
    {"Text": "Absolutely fantastic day!", "Emoji": "🎉", "Label": "Positive"},
    {"Text": "Why does this keep happening? ", "Emoji": "😩", "Label": "Negative"},
    {"Text": "It's just another average day.", "Emoji": "😐", "Label": "Neutral"},
    {"Text": "I'm so grateful for my friends! ", "Emoji": "🙏", "Label": "Positive"},
    {"Text": "Feeling overwhelmed with work.", "Emoji": "😣", "Label": "Negative"},
    {"Text": "Tonight was fun! ", "Emoji": "🎊", "Label": "Positive"},
    {"Text": "I'm not sure what to think about this.", "Emoji": "🤔", "Label": "Neutral"},
    {"Text": "Life is beautiful! ", "Emoji": "🌸", "Label": "Positive"},
    {"Text": "Just another disappointment.", "Emoji": "😞", "Label": "Negative"},
    {"Text": "Everything feels okay right now.", "Emoji": "😌", "Label": "Neutral"},
    {"Text": "This is amazing! ", "Emoji": "🎉", "Label": "Positive"},
    {"Text": "I'm really frustrated with this issue.", "Emoji": "😠", "Label": "Negative"},
    {"Text": "I just want to relax.", "Emoji": "🌙", "Label": "Neutral"},
    {"Text": "What a lovely surprise! ", "Emoji": "😊", "Label": "Positive"},
    {"Text": "I can't believe this is happening... ", "Emoji": "😱", "Label": "Negative"},
    {"Text": "Feeling a bit lost right now.", "Emoji": "😕", "Label": "Neutral"},
    {"Text": "I'm so happy with the results! ", "Emoji": "🎉", "Label": "Positive"},
    {"Text": "This is the worst experience ever! ", "Emoji": "😤", "Label": "Negative"},
    {"Text": "I'm neutral about this decision.", "Emoji": "😐", "Label": "Neutral"},
    {"Text": "Just finished a great book! ", "Emoji": "📚", "Label": "Positive"},
    {"Text": "Everything is going wrong today.", "Emoji": "😩", "Label": "Negative"},
    {"Text": "Just doing my best! ", "Emoji": "💪", "Label": "Neutral"},
    {"Text": "Today was absolutely amazing! ", "Emoji": "😍", "Label": "Positive"},
    {"Text": "I'm not enjoying this at all.", "Emoji": "😔", "Label": "Negative"},
    {"Text": "It's an average day for me.", "Emoji": "😐", "Label": "Neutral"},
]

# Specify the file path and name
csv_file = "dataset.csv"

# Writing to CSV file
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["Text", "Emoji", "Label"])
    writer.writeheader()
    writer.writerows(data)

print("CSV file created successfully!")
