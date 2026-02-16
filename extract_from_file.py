from esco_taxonomy import ESCOTaxonomy
import fitz  # PyMuPDF
import pdfplumber
import os
import re
import glob
import numpy as np

from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


# Lazy initialization of ESCO (avoid import-time failures)
_cache_dir = os.getenv('CACHE_DIR', '.')  # /app/cache on Render, . locally
_esco_instance = None

def get_esco():
    """Lazy load ESCO taxonomy on first use (avoids import-time failures)"""
    global _esco_instance
    if _esco_instance is None:
        _esco_instance = ESCOTaxonomy(
            csv_path="data/esco/skills_en.csv",
            cache_path=os.path.join(_cache_dir, "embedding_cache.pkl"),
            esco_cache_path=os.path.join(_cache_dir, "esco_embeddings.pkl")
        )
    return _esco_instance

esco = None  # Will be set on first use

# ==========================================================
# MODEL LOADING (LAZY - LOAD ON FIRST USE, NOT AT IMPORT TIME)
# ==========================================================

# These will be loaded lazily on first use
nlp = None
model = None

def _load_spacy_model():
    """Lazy load spaCy model on first use."""
    global nlp
    if nlp is not None:
        return nlp
    
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("Downloading spaCy model (first use)...")
        import subprocess
        try:
            result = subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"],
                capture_output=True,
                timeout=120
            )
            if result.returncode != 0:
                print(f"WARNING: spaCy download failed: {result.stderr.decode()}")
                return None
        except subprocess.TimeoutExpired:
            print("WARNING: spaCy download timed out")
            return None
        
        try:
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except Exception as e:
            print(f"WARNING: Could not load spaCy: {e}")
            return None

def _load_embedding_model():
    """Lazy load embedding model on first use."""
    global model
    if model is not None:
        return model
    
    try:
        print("Loading embedding model (first use)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✓ Embedding model loaded")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}")
        return None


# ==========================================================
# ATTACHED FILE EXTRACTION
# ==========================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_tables_from_pdf(pdf_path: str):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables.extend(page.extract_tables())
    return tables


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()


# ==========================================================
# NLP EXTRACTION
# ==========================================================

def extract_skills(text: str) -> set:
    """Extract skills using lazy-loaded spaCy model."""
    global nlp
    nlp = _load_spacy_model()
    skills = set()
    
    # If spaCy failed to load, use simple fallback
    if nlp is None:
        # Fallback: Simple noun extraction with regex
        words = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
        return set(words) if words else set()
    
    doc = nlp(text)

    for ent in doc.ents:
        skills.add(ent.text)

    for chunk in doc.noun_chunks:
        if len(chunk.text) > 2:
            skills.add(chunk.text)

    return skills


def extract_education(text: str) -> str:
    edu_keywords = ["Bachelor", "Master", "PhD", "BSc", "MSc", "Doctorate", "degree", "diploma"]
    for line in text.split('\n'):
        for kw in edu_keywords:
            if kw.lower() in line.lower():
                return line
    return ""


def extract_title(text: str) -> str:
    match = re.search(r'(Job Title|Position|Role)[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
    return match.group(2).strip() if match else ""


def extract_years_experience(text: str) -> int:
    matches = re.findall(r'(\d+)\+?\s+years?', text, re.IGNORECASE)
    return max(map(int, matches)) if matches else 0


# ==========================================================
# MATCHING LOGIC
# ==========================================================

def skill_match(cv_skills: set, jd_skills: set) -> float:
    if not cv_skills or not jd_skills:
        return 0.0

    cv_lower = {s.lower() for s in cv_skills}
    jd_lower = {s.lower() for s in jd_skills}

    exact = cv_lower & jd_lower
    exact_score = len(exact) / len(jd_lower)

    remaining_cv = cv_lower - exact
    remaining_jd = jd_lower - exact

    fuzzy_matches = 0
    for cv_skill in remaining_cv:
        for jd_skill in remaining_jd:
            if fuzz.ratio(cv_skill, jd_skill) > 85:
                fuzzy_matches += 1
                break

    fuzzy_score = fuzzy_matches / len(jd_lower)

    return 0.7 * exact_score + 0.3 * fuzzy_score


def experience_match(cv_text: str, jd_text: str) -> int:
    return int(
        extract_years_experience(cv_text) >=
        extract_years_experience(jd_text)
    )


def education_match(cv_edu: str, jd_edu: str) -> float:
    if not cv_edu or not jd_edu:
        return 0.0

    edu_levels = {
        'phd': 4,
        'doctorate': 4,
        'master': 3,
        'msc': 3,
        'bachelor': 2,
        'bsc': 2,
        'diploma': 1
    }

    cv_lower = cv_edu.lower()
    jd_lower = jd_edu.lower()

    cv_level = max((edu_levels.get(level, 0) for level in edu_levels if level in cv_lower), default=0)
    jd_level = max((edu_levels.get(level, 0) for level in edu_levels if level in jd_lower), default=0)

    base = fuzz.token_set_ratio(cv_edu, jd_edu) / 100

    if cv_level >= jd_level:
        base += 0.2
    else:
        base *= 0.8

    return min(1.0, base)


def title_match(cv_title: str, jd_title: str) -> float:
    return SequenceMatcher(None, cv_title.lower(), jd_title.lower()).ratio()


def tfidf_similarity(cv_text: str, jd_text: str) -> float:
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform([cv_text, jd_text])
    return cosine_similarity(matrix[0:1], matrix[1:2])[0][0]


def semantic_similarity(cv_text: str, jd_text: str) -> float:
    global model
    model = _load_embedding_model()

    if model is None:
        print("WARNING: Embedding model not available, using TF-IDF similarity instead")
        return tfidf_similarity(cv_text, jd_text)

    cv_emb = model.encode(cv_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    return util.cos_sim(cv_emb, jd_emb).item()


def extract_locations(text: str) -> set:
    """
    Extract GPE and LOC entities using lazy-loaded spaCy model.
    Falls back safely if model is unavailable.
    """
    global nlp
    nlp = _load_spacy_model()

    if nlp is None:
        return set()

    doc = nlp(text)

    return {
        ent.text.lower().strip()
        for ent in doc.ents
        if ent.label_ in {"GPE", "LOC"}
    }


def location_match(cv_text: str, jd_text: str) -> int:
    # Normalize text
    cv_text_lower = cv_text.lower()
    jd_text_lower = jd_text.lower()

    # 1️⃣ Remote handling
    if "remote" in jd_text_lower:
        return 1

    cv_locations = extract_locations(cv_text)
    jd_locations = extract_locations(jd_text)

    if not cv_locations or not jd_locations:
        return 0

    # 2️⃣ Exact match
    if cv_locations.intersection(jd_locations):
        return 1

    # 3️⃣ Simple hierarchical heuristic (city in country case)
    for cv_loc in cv_locations:
        for jd_loc in jd_locations:
            if cv_loc in jd_loc or jd_loc in cv_loc:
                return 1

    # 4️⃣ Semantic fallback (lazy-loaded embedding model)
    global model
    model = _load_embedding_model()

    if model is None:
        return 0  # fallback safely

    cv_vectors = model.encode(list(cv_locations))
    jd_vectors = model.encode(list(jd_locations))

    similarity_matrix = cosine_similarity(cv_vectors, jd_vectors)

    if np.max(similarity_matrix) >= 0.80:
        return 1

    return 0


# ==========================================================
# FINAL SCORING
# ==========================================================

def final_match_score(cv_data: dict, jd_data: dict) -> float:

    skill = skill_match(cv_data['skills'], jd_data['skills'])
    sem = semantic_similarity(cv_data['text'], jd_data['text'])
    tfidf = tfidf_similarity(cv_data['text'], jd_data['text'])
    edu = education_match(cv_data['education'], jd_data['education'])
    exp = experience_match(cv_data['text'], jd_data['text'])
    title = title_match(cv_data['title'], jd_data['title'])
    loc = location_match(cv_data['text'], jd_data['text'])

    score = (
        0.15 * skill +
        0.40 * sem +
        0.15 * tfidf +
        0.10 * edu +
        0.10 * exp +
        0.05 * title +
        0.05 * loc
    )

    return round(min(1.0, score), 3)


def match_report(cv_data: dict, jd_data: dict) -> dict:

    return {
        'Skill Match': skill_match(cv_data['skills'], jd_data['skills']),
        'Experience Match': experience_match(cv_data['text'], jd_data['text']),
        'Education Match': education_match(cv_data['education'], jd_data['education']),
        'Title Match': title_match(cv_data['title'], jd_data['title']),
        'TF-IDF Similarity': tfidf_similarity(cv_data['text'], jd_data['text']),
        'Semantic Similarity': semantic_similarity(cv_data['text'], jd_data['text']),
        'Location Match': location_match(cv_data['text'], jd_data['text']),
        'Final Score': final_match_score(cv_data, jd_data)
    }


# ==========================================================
# BATCH RANKING
# ==========================================================

def rank_cvs(jd_data: dict, cv_folder: str):

    ranking = []

    for cv_path in glob.glob(os.path.join(cv_folder, "*.pdf")):
        text = clean_text(extract_text_from_pdf(cv_path))

        raw_skills = extract_skills(text)
        normalized = get_esco().normalize(raw_skills)
        
        cv_data = {
            'text': text,
            'skills': set(normalized.values()),
            'title': extract_title(text),
            'education': extract_education(text)
        }

        score = final_match_score(cv_data, jd_data)
        report = match_report(cv_data, jd_data)

        ranking.append((cv_path, score, report))


    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking


# ==========================================================
# EXPLAINABILITY & SCORING BREAKDOWN
# ==========================================================

def generate_match_explanation(report, cv_data, jd_data, cv_text, jd_text):
    """
    Generate human-readable explanation of the match score.
    
    Returns a dictionary with clear breakdown of why the candidate matched.
    """
    
    # Calculate skill gaps
    matching_skills = cv_data['skills'] & jd_data['skills']
    missing_skills = jd_data['skills'] - cv_data['skills']
    extra_skills = cv_data['skills'] - jd_data['skills']
    
    cv_years = extract_years_experience(cv_text)
    jd_years = extract_years_experience(jd_text)
    
    explanation = {
        'overall_score': report.get('Final Score', 0),
        'interpretation': _get_score_interpretation(report.get('Final Score', 0)),
        'top_reasons': [],
        'gaps': [],
        'strengths': []
    }
    
    # Top Reasons (Components that contributed positively)
    if report.get('Skill Match', 0) >= 0.7:
        explanation['top_reasons'].append({
            'icon': '✅',
            'metric': 'Skill Match',
            'score': f"{report['Skill Match']:.0%}",
            'detail': f"Found {len(matching_skills)} matching skills: {', '.join(list(matching_skills)[:5])}"
        })
    
    if report.get('Semantic Similarity', 0) >= 0.7:
        explanation['top_reasons'].append({
            'icon': '✅',
            'metric': 'Text Similarity',
            'score': f"{report['Semantic Similarity']:.0%}",
            'detail': 'Strong alignment between CV and job description content'
        })
    
    if report.get('Experience Match', 0) == 1:
        explanation['top_reasons'].append({
            'icon': '✅',
            'metric': 'Experience',
            'score': '100%',
            'detail': f"{cv_years} years provided ≥ {jd_years} years required"
        })
    
    if report.get('Education Match', 0) >= 0.7:
        explanation['top_reasons'].append({
            'icon': '✅',
            'metric': 'Education',
            'score': f"{report['Education Match']:.0%}",
            'detail': f"CV: {cv_data.get('education', 'Not found')} | JD: {jd_data.get('education', 'Not found')}"
        })
    
    if report.get('Title Match', 0) >= 0.6:
        explanation['top_reasons'].append({
            'icon': '✅',
            'metric': 'Title Match',
            'score': f"{report['Title Match']:.0%}",
            'detail': f"CV: {cv_data.get('title', 'N/A')} matches JD role requirements"
        })
    
    # Gaps (Areas needing improvement)
    if missing_skills:
        skills_str = ', '.join(list(missing_skills)[:5])
        explanation['gaps'].append({
            'icon': '⚠️',
            'type': 'Missing Skills',
            'detail': f"Job requires: {skills_str}"
        })
    
    if report.get('Skill Match', 0) < 0.5:
        explanation['gaps'].append({
            'icon': '⚠️',
            'type': 'Skill Gap',
            'detail': f"Only {report['Skill Match']:.0%} of required skills matched"
        })
    
    if cv_years < jd_years:
        explanation['gaps'].append({
            'icon': '⚠️',
            'type': 'Experience Gap',
            'detail': f"{cv_years} years provided, but {jd_years} years required"
        })
    
    if report.get('Education Match', 0) < 0.6:
        explanation['gaps'].append({
            'icon': '⚠️',
            'type': 'Education Mismatch',
            'detail': f"Education level only {report['Education Match']:.0%} aligned"
        })
    
    # Strengths (Nice to haves that help)
    if extra_skills:
        skills_str = ', '.join(list(extra_skills)[:5])
        explanation['strengths'].append({
            'icon': '⭐',
            'detail': f"Extra relevant skills: {skills_str}"
        })
    
    if report.get('TF-IDF Similarity', 0) >= 0.7:
        explanation['strengths'].append({
            'icon': '⭐',
            'detail': 'Strong terminology alignment with job description'
        })
    
    if cv_years > jd_years:
        explanation['strengths'].append({
            'icon': '⭐',
            'detail': f"Exceeds experience requirement by {cv_years - jd_years} years"
        })
    
    return explanation


def _get_score_interpretation(score):
    """Map score to human-readable interpretation."""
    if score >= 0.85:
        return ("🟢 Excellent Match", "Outstanding alignment with all requirements")
    elif score >= 0.75:
        return ("🟢 Strong Match", "Very good fit for the role")
    elif score >= 0.60:
        return ("🟡 Good Match", "Meets most requirements")
    elif score >= 0.45:
        return ("🟠 Moderate Match", "Some gaps but trainable")
    elif score >= 0.30:
        return ("🔴 Weak Match", "Significant gaps exist")
    else:
        return ("🔴 Poor Match", "Major mismatch with requirements")
