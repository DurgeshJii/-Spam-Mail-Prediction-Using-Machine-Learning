# ================================
# Spam Mail Prediction Using ML
# ================================

# Importing Dependencies
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# ================================
# Data Collection & Preprocessing
# ================================

# Load dataset
raw_mail_data = pd.read_csv('mail_data.csv')

# Replace null values with empty string
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

# Label Encoding
# spam -> 0 , ham -> 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate features and labels
X = mail_data['Message']
Y = mail_data['Category']

# Ensure correct data types
X = X.fillna("").astype(str)
Y = pd.Series(Y).fillna(0).infer_objects(copy=False).astype(int)


# ================================
# Train-Test Split
# ================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3
)


# ================================
# Feature Extraction (TF-IDF)
# ================================

vectorizer = TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True
)

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)


# ================================
# Model Training (Logistic Regression)
# ================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_features, Y_train)


# ================================
# Model Evaluation
# ================================

# Training accuracy
train_predictions = model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, train_predictions)

print("Accuracy on training data :", train_accuracy)

# Test accuracy
test_predictions = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_predictions)

print("Accuracy on test data :", test_accuracy)


# ================================
# Predictive System
# ================================

input_mail = [
    "I've been searching for the right words to thank you for this breather..."
]

input_data_features = vectorizer.transform(input_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")
