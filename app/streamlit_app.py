import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
import sys

# Backend setup preserved
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.parsing.resume_parser import extract_resume_text
from src.models.classifier import predict_category
from src.models.recommender import recommend_jobs
from src.models.fit_predictor import predict_fit

st.set_page_config(page_title="SmartHire AI", page_icon="🧭", layout="wide")

# Load the stylesheet — try the common places it might live relative to this
# script or the project root, so this works regardless of exact folder layout.
_APP_DIR = Path(__file__).resolve().parent
_CSS_CANDIDATES = [
    _APP_DIR / "style.css",
    _APP_DIR / "static" / "style.css",
    _APP_DIR / "assets" / "style.css",
    PROJECT_ROOT / "style.css",
    PROJECT_ROOT / "frontend" / "style.css",
    PROJECT_ROOT / "static" / "style.css",
]
_css_path = next((p for p in _CSS_CANDIDATES if p.exists()), None)
if _css_path:
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)
else:
    st.error(
        "style.css was not found next to streamlit_app.py (or in the usual "
        "static/assets folders), so the page is rendering with no styling. "
        "Move style.css into the same folder as streamlit_app.py, or edit "
        "the _CSS_CANDIDATES list above to point at wherever you keep it."
    )

# Brand row
st.markdown("""
    <div class="brand-row">
        <div class="brand-mark">S</div>
        <div class="brand-name">SmartHire AI</div>
    </div>
""", unsafe_allow_html=True)

# Center the Menu and Style it to look like a Navbar
# We remove the sidebar entirely.
selected = option_menu(
    menu_title=None,
    options=["Home", "Resume", "Jobs", "Fit"],
    icons=["house", "file-earmark-text", "briefcase", "graph-up"],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "6px!important",
            "background-color": "#FFFFFF",
            "max-width": "640px",
            "margin": "0 auto",
            "border-radius": "999px",
            "border": "1px solid #E6E9F2",
            "box-shadow": "0 2px 10px rgba(16,27,51,0.06)",
        },
        "icon": {"color": "#5C6478", "font-size": "15px"},
        "nav-link": {
            "font-family": "Inter",
            "font-size": "15px",
            "font-weight": "600",
            "text-align": "center",
            "margin": "0px",
            "padding": "10px 20px",
            "border-radius": "999px",
            "color": "#5C6478",
        },
        "nav-link-selected": {"background-color": "#101B33", "color": "#FFFFFF"},
    }
)

# Hero Section
if selected == "Home":
    st.markdown("""
        <div class="hero-wrap">
            <div class="match-ring-floating">
                <div class="ring">
                    <div class="ring-inner">
                        <div class="ring-value">92%</div>
                        <div class="ring-label">MATCH</div>
                    </div>
                </div>
                <div class="ring-caption">Sample fit score</div>
            </div>
            <div class="hero-section">
                <div class="eyebrow">AI-POWERED CAREER MATCHING</div>
                <h1>Modernizing the<br>Job Search Experience</h1>
                <p class="hero-sub">
                    SmartHire AI bridges the gap between your unique skills and the perfect career
                    opportunity — powered by resume parsing, smart matching, and fit scoring.
                </p>
                <div class="pill-row">
                    <span class="pill">📄 Resume Parsing</span>
                    <span class="pill">💼 Smart Matching</span>
                    <span class="pill">📊 Fit Scoring</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-eyebrow">The Process</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="steps-row">
            <div class="step-card">
                <div class="step-number">01</div>
                <h4>Upload your resume</h4>
                <p>Drop in a PDF or Word file and let SmartHire read and structure your experience.</p>
            </div>
            <div class="step-card">
                <div class="step-number">02</div>
                <h4>We analyze it</h4>
                <p>NLP extracts your skills and classifies your professional category automatically.</p>
            </div>
            <div class="step-card">
                <div class="step-number">03</div>
                <h4>Get matched</h4>
                <p>See ranked job recommendations and a fit score against any job description.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

    # Larger, custom-styled cards
    st.markdown("""
        <div style="display: flex; gap: 20px; justify-content: center; margin-top: 20px;">
            <div class="custom-card">
                <div class="icon-badge navy">📄</div>
                <h3>Resume Analysis</h3>
                <p>Upload your resume to extract key information. We use NLP to parse your documents and automatically classify your professional role/category.</p>
            </div>
            <div class="custom-card">
                <div class="icon-badge coral">💼</div>
                <h3>Job Recommendations</h3>
                <p>Discover tailored job openings. Our engine calculates cosine similarity between your resume and thousands of listings to find the best match.</p>
            </div>
            <div class="custom-card">
                <div class="icon-badge teal">📊</div>
                <h3>Fit Analysis</h3>
                <p>Not sure if you're a good fit? Paste any job description to get a precise matching score and identify potential skill gaps in your profile.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
# ... (Keep all your existing elif blocks for Resume, Jobs, Fit below this)
# --- ALL YOUR OTHER ELIF BLOCKS REMAIN IDENTICAL (backend logic untouched) ---
elif selected == "Resume":
    st.markdown("""
        <div class="page-header">
            <div class="page-icon">📄</div>
            <div>
                <h2>Resume Analysis</h2>
                <p>Upload a resume to extract key details and classify your professional category.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose Resume", type=["pdf", "docx"])
    if uploaded_file is not None:
        temp_dir = PROJECT_ROOT / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / uploaded_file.name
        with open(temp_file, "wb") as f: f.write(uploaded_file.getbuffer())
        with st.spinner("Analyzing Resume..."):
            resume_text = extract_resume_text(temp_file)
            category = predict_category(resume_text)
        st.session_state["resume_text"] = resume_text
        st.session_state["resume_category"] = category

        st.markdown('<div class="banner-success">✅ Resume analyzed successfully!</div>', unsafe_allow_html=True)

        left, right = st.columns([1, 3])
        with left:
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Category</div>
                    <div class="result-value">{category}</div>
                </div>
            """, unsafe_allow_html=True)
        with right:
            st.markdown('<p style="font-weight:600; color:#101B33; margin-bottom:6px;">Extracted Resume</p>', unsafe_allow_html=True)
            st.text_area("Extracted Resume", resume_text, height=350, label_visibility="collapsed")

    elif "resume_text" in st.session_state:
        st.markdown(f"""
            <div class="banner-success">
                📌 A resume is still on file (category: {st.session_state.get('resume_category', 'N/A')}).
                Jobs and Fit will keep using it until you upload a new one below.
            </div>
        """, unsafe_allow_html=True)
        if st.button("🗑 Clear analyzed resume"):
            st.session_state.pop("resume_text", None)
            st.session_state.pop("resume_category", None)
            st.rerun()

elif selected == "Jobs":
    st.markdown("""
        <div class="page-header">
            <div class="page-icon">💼</div>
            <div>
                <h2>Job Recommendations</h2>
                <p>Ranked openings based on similarity to your analyzed resume.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if "resume_text" not in st.session_state:
        st.markdown('<div class="banner-warning">⚠️ Please analyze a resume first.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Finding matching jobs..."):
            jobs = recommend_jobs(st.session_state["resume_text"], top_n=5)
        jobs["Similarity"] = (jobs["Similarity"] * 100).round(2)
        for _, job in jobs.iterrows():
            st.markdown(f"""
                <div class="job-card">
                    <div class="job-card-top">
                        <div>
                            <p class="job-title">{job['title']}</p>
                            <p class="job-meta">🏢 {job['company']} &nbsp;|&nbsp; 📍 {job['location']}</p>
                        </div>
                        <div class="match-badge">{job['Similarity']}% match</div>
                    </div>
                    <div class="match-bar-track">
                        <div class="match-bar-fill" style="width:{job['Similarity']}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

elif selected == "Fit":
    st.markdown("""
        <div class="page-header">
            <div class="page-icon">📊</div>
            <div>
                <h2>Resume Fit</h2>
                <p>Paste a job description to see how well your resume matches it.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if "resume_text" not in st.session_state:
        st.markdown('<div class="banner-warning">⚠️ Please analyze a resume first.</div>', unsafe_allow_html=True)
    else:
        job_description = st.text_area("Paste Job Description", height=220)
        if st.button("Calculate Fit Score"):
            if job_description.strip() == "":
                st.markdown('<div class="banner-warning">⚠️ Please enter a job description.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Calculating..."):
                    score, level = predict_fit(st.session_state["resume_text"], job_description)

                st.markdown(f"""
                    <div class="fit-result-row">
                        <div class="fit-ring" style="background: conic-gradient(#12A594 0% {score}%, #E6E9F2 {score}% 100%);">
                            <div class="fit-ring-inner">{score}%</div>
                        </div>
                        <div>
                            <div class="fit-level-label">Overall Level</div>
                            <div class="fit-level-value">{level}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)