
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

df = pd.read_csv("sentiment_dataset.csv")
df = df[["text", "sentiment"]].dropna()

df["text"] = df["text"].astype(str)
df["sentiment"] = df["sentiment"].str.lower().str.strip()

df = df[df["sentiment"].isin(
    ["positive", "negative", "neutral"]
)]

print("Total examples:", len(df))
print(df["sentiment"].value_counts())

X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTraining completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/sentiment_model.pkl")

print("\nNew three-class model saved successfully!")
