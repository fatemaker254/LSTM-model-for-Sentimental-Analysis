import pandas as pd
import numpy as np
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

# Ensure consistent number of samples
min_samples = min(len(y_test), len(y_pred))
y_test = y_test.iloc[:min_samples]
y_pred = y_pred[:min_samples]

# Convert labels in y_test to integer format
label_mapping = {'Negative': 0, 'Positive': 1}  # Adjust this based on your actual labels
y_test = y_test.map(label_mapping).astype(int)

# Check Unique Values
print("Unique values in y_test:", np.unique(y_test))
print("Unique values in y_pred:", np.unique(y_pred))

# Verify Data Types
print("Data type of y_test:", y_test.dtype)
print("Data type of y_pred:", y_pred.dtype)

# Label Mapping
print("Label Mapping:", label_mapping)

# Handling Unknown Labels
unknown_labels_y_test = set(np.unique(y_test)) - set(np.unique(y_pred))
unknown_labels_y_pred = set(np.unique(y_pred)) - set(np.unique(y_test))

if unknown_labels_y_test or unknown_labels_y_pred:
    # Implement your strategy for handling unknown labels
    print("Unknown labels in y_test:", unknown_labels_y_test)
    print("Unknown labels in y_pred:", unknown_labels_y_pred)

# Check if there is a mix of binary and unknown targets
if unknown_labels_y_test and unknown_labels_y_pred:
    raise ValueError("Mix of binary and unknown targets in the labels.")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report_str)
