This project focuses on emotion detection using text and emojis, utilizing both traditional machine learning and deep learning algorithms. The objective is to classify emotions into Positive, Neutral, and Negative categories, based on a custom-built dataset.

Dataset
Since no suitable dataset was available online, a custom dataset was created. The dataset contains the following fields:

Text: Sentences or phrases representing user emotions.
Emoji: Emojis associated with the text to provide context.
Label: Emotion category, one of Positive, Neutral, or Negative.
The dataset was carefully curated to cover a wide range of expressions, including text and emoji combinations.

Algorithms
The project implements the following algorithms:

Naive Bayes
Logistic Regression
Decision Tree
LSTM (Long Short-Term Memory)
Evaluation
Each model was evaluated using the following metrics:

Accuracy
Precision
Recall
F1 Score
Additionally, confusion matrices were plotted for detailed performance analysis. A 5-fold cross-validation approach was used to ensure robust evaluation and mitigate overfitting.

Project Features
Custom Dataset: A unique dataset containing text, emojis, and labels for emotion classification.
Preprocessing: Handled text and emoji data effectively, ensuring compatibility for both traditional and deep learning models.
Algorithms: Compared the performance of machine learning models (Naive Bayes, Logistic Regression, Decision Tree) and an LSTM deep learning model.
Metrics: Comprehensive evaluation using multiple metrics and confusion matrices.
Cross-validation: Ensured model reliability using 5-fold cross-validation.
Results
The project compared the performance of all four algorithms on the custom dataset. Detailed analysis of results and visualizations, including confusion matrices, are provided in the results section of the project.
