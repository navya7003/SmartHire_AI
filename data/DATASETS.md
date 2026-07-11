# Datasets

SmartHire AI uses two datasets to perform resume classification, job recommendation, and exploratory job clustering.

---

## 1. Resume Dataset

Purpose:
- Train the resume classification model.

Details:
- Total resumes: 962
- Categories: 25
- File format: CSV
- Used for supervised learning (Logistic Regression)

Output:
- Predicts the professional category of an uploaded resume.

---

## 2. Job Postings Dataset

Purpose:
- Recommend relevant jobs.
- Calculate resume fit score.
- Perform exploratory job clustering.

Details:
- Total job postings: 136,759
- Includes job title, company, location, experience, skills, and description.

Applications:
- Content-based Job Recommendation
- Resume Fit Prediction
- K-Means Job Clustering

---

## Data Processing

The datasets are preprocessed before training:

- Missing value handling
- Text cleaning
- TF-IDF Vectorization
- Feature extraction

---

## Note

The datasets are provided for educational and research purposes. Please ensure you have the appropriate permissions before using them in other projects.