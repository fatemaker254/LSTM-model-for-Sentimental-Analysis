import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("data/sentiment_data.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data["Text"], data["Sentiment"], test_size=0.2, random_state=42
)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Save the vectorizer and processed data
pd.DataFrame(
    X_train_vectorized.toarray(), columns=vectorizer.get_feature_names_out()
).to_csv("data/X_train_vectorized.csv", index=False)
pd.DataFrame(
    X_test_vectorized.toarray(), columns=vectorizer.get_feature_names_out()
).to_csv("data/X_test_vectorized.csv", index=False)
pd.DataFrame({"y_train": y_train, "y_test": y_test}).to_csv(
    "data/labels.csv", index=False
)
