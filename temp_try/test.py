# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv("adult.csv")  # adjust the path if needed
print("Initial data shape:", df.shape)
display(df.head())

# 2. Clean and preprocess the data
# Strip whitespace from object type columns
for col in df.select_dtypes(['object']).columns:
    df[col] = df[col].str.strip()

# Replace '?' values with NaN and drop such rows
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
print("After removing missing:", df.shape)

# Optionally drop columns that are not needed
if 'education' in df.columns:
    df.drop('education', axis=1, inplace=True)  # 'education' is redundant if 'educational-num' exists

# 3. Encode categorical variables
cat_cols = df.select_dtypes(include=["object"]).columns.drop(['income'])
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # store encoder for each column

# Encode the target
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# Final feature and target separation
X = df.drop('income', axis=1)
y = df['income']

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Fit a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Predict and evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
