from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

classifier = joblib.load(
    MODEL_DIR / "resume_classifier.pkl"
)

tfidf = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

label_encoder = joblib.load(
    MODEL_DIR / "label_encoder.pkl"
)

def predict_category(resume_text):

    resume_vector = tfidf.transform([resume_text])

    prediction = classifier.predict(resume_vector)[0]

    category = label_encoder.inverse_transform(
        [prediction]
    )[0]

    return category