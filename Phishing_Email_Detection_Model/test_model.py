import joblib
from pathlib import Path

MODEL_FILE = Path(__file__).resolve().parent / "models" / "phishing_model.joblib"

if not MODEL_FILE.exists():
    print("Train the model first: python train_model.py")
    raise SystemExit(1)

model = joblib.load(MODEL_FILE)

samples = [
    "Urgent security alert. Verify your password immediately by clicking this link.",
    "Hi team, our project meeting is scheduled for tomorrow at 10 AM."
]

for text in samples:
    prediction = model.predict([text])[0]
    probability = max(model.predict_proba([text])[0])
    print(f"\nEmail: {text}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {probability:.2%}")
