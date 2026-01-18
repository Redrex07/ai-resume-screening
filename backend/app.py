from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------------------------------
# APP SETUP
# ---------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------
# SKILL WEIGHTS (AI INTELLIGENCE)
# ---------------------------------
SKILL_WEIGHTS = {
    "python": 5,
    "machine learning": 5,
    "ai": 5,
    "flask": 4,
    "django": 4,
    "sql": 3,
    "mysql": 3,
    "html": 2,
    "css": 2,
    "javascript": 2,
    "excel": 1,
    "azure": 3,
    "aws": 3,
    "docker": 3,
    "git": 2
}

SKILLS_DB = list(SKILL_WEIGHTS.keys())

# ---------------------------------
# TEXT CLEANING
# ---------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text

# ---------------------------------
# FILE TEXT EXTRACTION
# ---------------------------------
def extract_text(file):
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join(page.extract_text() or "" for page in reader.pages)

    if file.filename.endswith(".docx"):
        doc = Document(file)
        return " ".join(p.text for p in doc.paragraphs)

    return ""

# ---------------------------------
# NLP SIMILARITY (SUPPORTING)
# ---------------------------------
def text_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return int(score * 100)

# ---------------------------------
# ATS WEIGHTED SCORE (MAIN)
# ---------------------------------
def weighted_skill_score(matched, missing):
    matched_score = sum(SKILL_WEIGHTS[s] for s in matched)
    missing_score = sum(SKILL_WEIGHTS[s] for s in missing)

    total = matched_score + missing_score
    if total == 0:
        return 0

    return int((matched_score / total) * 100)

# ---------------------------------
# AI SUMMARY
# ---------------------------------
def resume_summary(text):
    sentences = text.split(".")
    return ". ".join(sentences[:3]).strip() + "."

# ---------------------------------
# DECISION ENGINE
# ---------------------------------
def decision_engine(score):
    if score >= 75:
        return "Strong Fit", "High", "Shortlist"
    elif score >= 55:
        return "Moderate Fit", "Medium", "Review"
    else:
        return "Low Fit", "Low", "Reject"

# ---------------------------------
# AI ANALYZE ROUTE (PHASE X)
# ---------------------------------
@app.route("/api/ai-analyze", methods=["POST"])
def ai_analyze():
    resume = request.files.get("resume")
    jd = request.form.get("jd")

    if not resume or not jd:
        return jsonify({"error": "Resume or Job Description missing"}), 400

    resume_text = clean_text(extract_text(resume))
    jd_text = clean_text(jd)

    matched = [s for s in SKILLS_DB if s in resume_text and s in jd_text]
    missing = [s for s in SKILLS_DB if s in jd_text and s not in resume_text]

    ats_score = weighted_skill_score(matched, missing)
    nlp_score = text_similarity(resume_text, jd_text)

    final_score = int((ats_score * 0.7) + (nlp_score * 0.3))

    decision, confidence, verdict = decision_engine(final_score)

    suggestions = missing[:5]

    return jsonify({
        "match_score": final_score,
        "ats_score": ats_score,
        "nlp_similarity": nlp_score,
        "decision": decision,
        "confidence": confidence,
        "verdict": verdict,
        "matched_skills": matched,
        "missing_skills": missing,
        "improvement_suggestions": suggestions,
        "resume_summary": resume_summary(resume_text)
    })

# ---------------------------------
# RUN SERVER
# ---------------------------------
if __name__ == "__main__":
    app.run()


