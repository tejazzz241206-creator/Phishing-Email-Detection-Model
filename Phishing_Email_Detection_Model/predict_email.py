import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "phishing_model.joblib"

def predict_email(subject, body):
    model = joblib.load(MODEL_FILE)
    text = subject + " " + body

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    classes = list(model.classes_)
    confidence = probabilities[classes.index(prediction)]

    return prediction, confidence

print("=== PHISHING EMAIL DETECTOR ===")

if not MODEL_FILE.exists():
    print("\nModel not found.")
    print("Run this first: python train_model.py")
    raise SystemExit(1)

subject = input("\nEnter email subject: ").strip()
body = input("Enter email body: ").strip()

prediction, confidence = predict_email(subject, body)

print("\n--- RESULT ---")
print(f"Prediction : {prediction.upper()}")
print(f"Confidence : {confidence:.2%}")

if prediction == "phishing":
    print("Warning: This email shows characteristics commonly associated with phishing.")
    print("Do not click links, open unexpected attachments, or provide passwords.")
else:
    print("This email is classified as legitimate by the model.")
    print("Still verify unexpected requests independently.")
