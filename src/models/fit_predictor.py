"""Optional supervised shortlisting / fit predictor."""
"""
============================================================
SmartHire AI/ML Project
Resume Fit Prediction Module
============================================================

Calculates the similarity between a resume and a job description.
"""

from pathlib import Path

import joblib

from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Load TF-IDF Vectorizer
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

tfidf = joblib.load(
    MODEL_DIR / "job_tfidf_vectorizer.pkl"
)


# ============================================================
# Predict Resume Fit
# ============================================================

def predict_fit(resume_text, job_description):
    """
    Calculates the resume-job match percentage.
    """

    # Vectorize resume
    resume_vector = tfidf.transform([resume_text])

    # Vectorize job
    job_vector = tfidf.transform([job_description])

    # Cosine similarity
    similarity = cosine_similarity(
        resume_vector,
        job_vector
    )[0][0]

    # Convert to percentage
    fit_score = similarity * 100

    # Determine fit level
    if fit_score >= 80:
        fit_level = "Excellent Match"

    elif fit_score >= 60:
        fit_level = "Good Match"

    elif fit_score >= 40:
        fit_level = "Moderate Match"

    else:
        fit_level = "Low Match"

    return round(fit_score, 2), fit_level