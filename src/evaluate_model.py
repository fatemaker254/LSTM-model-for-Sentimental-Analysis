import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessed data
X_test_vectorized = pd.read_csv("data/X_test_vectorized.csv")
y_test = pd.read_csv("data/labels.csv")["y_test"]

# Load the trained model
model = joblib.load("models/sentiment_model.pkl")

# Drop rows with 'nan' labels in the test set
nan_indices = y_test.index[y_test.isna()]
X_test_vectorized = X_test_vectorized.drop(index=nan_indices, errors='ignore')
y_test = y_test.drop(index=nan_indices)

# Check for missing values in X_test_vectorized
if X_test_vectorized.isnull().values.any():
    # Handle missing values by dropping rows or using another strategy
    X_test_vectorized = X_test_vectorized.dropna()  # You can choose a different strategy based on your data

# Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report_str)
