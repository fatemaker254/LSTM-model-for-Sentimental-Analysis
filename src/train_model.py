import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib

# Load preprocessed data
X_train_vectorized = pd.read_csv("data/X_train_vectorized.csv")
y_train = pd.read_csv("data/labels.csv")["y_train"]

# Drop rows with 'nan' labels
nan_indices = y_train.index[y_train.isna()]
X_train_vectorized = X_train_vectorized.drop(index=nan_indices, errors='ignore')
y_train = y_train.drop(index=nan_indices)

# Check for missing values in X_train_vectorized
if X_train_vectorized.isnull().values.any():
    # Handle missing values by dropping rows or using another strategy
    X_train_vectorized = X_train_vectorized.dropna()  # You can choose a different strategy based on your data

# Encode labels as integers
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Ensure consistent number of samples
min_samples = min(len(X_train_vectorized), len(y_train_encoded))
X_train_vectorized = X_train_vectorized.iloc[:min_samples, :]
y_train_encoded = y_train_encoded[:min_samples]

# Check for indices and duplicates
print("Indices of X_train_vectorized:", X_train_vectorized.index)
print("Indices of y_train_encoded:", range(len(y_train_encoded)))

print("Number of duplicates in X_train_vectorized:", X_train_vectorized.duplicated().sum())
print("Number of duplicates in y_train_encoded:", pd.Series(y_train_encoded).duplicated().sum())

# Train a simple sentiment analysis model (Naive Bayes as an example)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train_encoded)

# Save the trained model and label encoder
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
