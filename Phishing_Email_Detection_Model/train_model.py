import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "emails.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_FILE)

# Combine subject and body into one text field
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

X = df["text"]
y = df["label"]

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 3. TF-IDF + Logistic Regression pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 4. Train
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)

print("\n=== PHISHING EMAIL DETECTION MODEL ===")
print(f"Dataset size: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"\nAccuracy: {accuracy_score(y_test, predictions):.2%}")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 6. Save the complete pipeline
model_path = MODEL_DIR / "phishing_model.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")
print("Training completed successfully.")
