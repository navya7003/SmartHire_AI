"""Unsupervised content-based recommender: cosine-similarity job ranking (top-N)."""
"""
============================================================
SmartHire AI/ML Project
Job Recommendation Module
============================================================

Unsupervised content-based recommender:
Ranks jobs using TF-IDF + Cosine Similarity.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Load Dataset and TF-IDF Vectorizer
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

jobs_df = pd.read_csv(DATA_DIR / "jobs_clean.csv")

tfidf = joblib.load(
    MODEL_DIR / "job_tfidf_vectorizer.pkl"
)

job_vectors = tfidf.transform(
    jobs_df["text"]
)
# ============================================================
# Recommend Jobs
# ============================================================

def recommend_jobs(resume_text, top_n=5):
    """
    Recommend the top matching jobs for a resume.
    """

    # Convert resume to TF-IDF
    resume_vector = tfidf.transform([resume_text])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(
        resume_vector,
        job_vectors
    ).flatten()

    # Get indices of top matches
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    # Select jobs
    recommendations = jobs_df.iloc[top_indices].copy()

    # Add similarity score
    recommendations["Similarity"] = similarity_scores[top_indices]

    return recommendations