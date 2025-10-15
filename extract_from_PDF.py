
import fitz  # PyMuPDF
import pdfplumber
import layoutparser as lp
import os
import numpy as np   # ✅ Add this line

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_images_from_pdf(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"page{page_num}_img{img_index}.{image_ext}"
                with open(os.path.join(output_dir, image_filename), "wb") as f:
                    f.write(image_bytes)

def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            tables.extend(page_tables)
    return tables

def extract_layout_blocks(pdf_path):
    layout_blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            image = page.to_image(resolution=150)
            img = image.original
            model = lp.models.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')
            layout = model.detect(img)
            layout_blocks.append(layout)
    return layout_blocks

# Example usage:
pdf_file = "ivan_resume.pdf"
cv_text = extract_text_from_pdf(pdf_file)
print("Resume Preview:\n", cv_text[:500])  # Print first 500 characters

# Extract images
extract_images_from_pdf(pdf_file, "extracted_images")
print("Images extracted to 'extracted_images' directory.")

# Extract tables
tables = extract_tables_from_pdf(pdf_file)
if tables:
    print(f"\nExtracted {len(tables)} tables from resume PDF.")
    for i, table in enumerate(tables):
        print(f"Table {i+1}:")
        for row in table:
            print(row)
else:
   print("\nNo tables found in resume PDF.")

# Extract layout blocks (optional, can be slow)
try:
    layout_blocks = extract_layout_blocks(pdf_file)
    print(f"\nExtracted layout blocks from {pdf_file} (first page):")
    print(layout_blocks[0])
except Exception as e:
   print(f"\nLayout extraction failed: {e}")


# Extract text
jd_file = "sample-job-description.pdf"
jd_text = extract_text_from_pdf(jd_file)
print("Job Description Preview:\n", jd_text)  # Print first 500 characters

# Extract images
extract_images_from_pdf(jd_file, "jd_extracted_images")
print("Images extracted from job description to 'jd_extracted_images' directory.")

# Extract tables
jd_tables = extract_tables_from_pdf(jd_file)
if jd_tables:
    print(f"\nExtracted {len(jd_tables)} tables from job description PDF.")
    for i, table in enumerate(jd_tables):
        print(f"Table {i+1}:")
        for row in table:
            print(row)
else:
    print("\nNo tables found in job description PDF.")


# Extract layout blocks (optional, can be slow)
try:
    jd_layout_blocks = extract_layout_blocks(jd_file)
    print(f"\nExtracted layout blocks from {jd_file} (first page):")
    print(jd_layout_blocks[0])
except Exception as e:
    print(f"\nLayout extraction for job description failed: {e}")

# --- Semantic Similarity Comparison ---
from sentence_transformers import SentenceTransformer, util
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove newlines and extra spaces
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII
    return text.strip()

cv_text_clean = clean_text(cv_text)
jd_text_clean = clean_text(jd_text)

# Load the model (small, fast, and accurate)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode both texts
cv_embedding = model.encode(cv_text_clean, convert_to_tensor=True)
jd_embedding = model.encode(jd_text_clean, convert_to_tensor=True)

# Compute cosine similarity
similarity_score = util.cos_sim(cv_embedding, jd_embedding).item()

print(f"\n🔍 Similarity Score between CV and Job Description: {similarity_score:.2f}")

print("\n📊 Step 2.3: What Does the Score Mean?")
print("Score Range\tInterpretation")
print("0.80 – 1.00\tVery strong match")
print("0.60 – 0.79\tGood match, potentially relevant")
print("0.40 – 0.59\tMatch, needs review")

print("< 0.40\tLikely irrelevant or very generic/resume mismatch")

# --- Keyword Extraction / Named Entity Recognition (NER) ---
import spacy

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_skills(text):
    doc = nlp(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ in ["SKILL", "ORG", "PERSON", "NORP", "GPE", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE"]:
            skills.add(ent.text)
    # Also add noun chunks as potential skills
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 2:
            skills.add(chunk.text)
    return skills

cv_skills = extract_skills(cv_text_clean)
jd_skills = extract_skills(jd_text_clean)

matched_skills = cv_skills.intersection(jd_skills)
missing_skills = jd_skills.difference(cv_skills)

print(f"\n🧩 Skills/Keywords in Job Description: {sorted(jd_skills)}")
print(f"\n🧩 Skills/Keywords in CV: {sorted(cv_skills)}")

print(f"\n✅ Skills from JD found in CV: {sorted(matched_skills)}")
print(f"\n❌ Skills in JD missing from CV: {sorted(missing_skills)}")


# --- Modular Scoring Functions ---
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util

# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def skill_match(cv_skills: set, jd_skills: set) -> float:
    if not cv_skills or not jd_skills:
        return 0.0
    
    # Convert skills to lowercase for better matching
    cv_skills_lower = {s.lower() for s in cv_skills}
    jd_skills_lower = {s.lower() for s in jd_skills}
    
    # Calculate exact matches
    exact_matches = cv_skills_lower & jd_skills_lower
    exact_score = len(exact_matches) / len(jd_skills_lower) if jd_skills_lower else 0
    
    # Calculate fuzzy matches for remaining skills
    remaining_cv = cv_skills_lower - exact_matches
    remaining_jd = jd_skills_lower - exact_matches
    fuzzy_matches = 0
    
    for cv_skill in remaining_cv:
        for jd_skill in remaining_jd:
            if fuzz.ratio(cv_skill, jd_skill) > 80:  # High similarity threshold
                fuzzy_matches += 1
                break
    
    fuzzy_score = fuzzy_matches / len(jd_skills_lower) if jd_skills_lower else 0
    
    # Combine scores with more weight on exact matches
    return 0.7 * exact_score + 0.3 * fuzzy_score

def extract_years_experience(text: str) -> int:
    matches = re.findall(r'(\d+)\+?\s+years?', text, re.IGNORECASE)
    return max(map(int, matches)) if matches else 0

def experience_match(cv_text: str, jd_text: str) -> int:
    cv_years = extract_years_experience(cv_text)
    jd_years = extract_years_experience(jd_text)
    return int(cv_years >= jd_years)

def education_match(cv_edu: str, jd_edu: str) -> float:
    if not cv_edu or not jd_edu:
        return 0.0
    
    # Define education levels and their relative weights
    edu_levels = {
        'phd': 4,
        'doctorate': 4,
        'master': 3,
        'msc': 3,
        'bachelor': 2,
        'bsc': 2,
        'diploma': 1
    }
    
    cv_edu_lower = cv_edu.lower()
    jd_edu_lower = jd_edu.lower()
    
    # Find the highest education level mentioned in each
    cv_level = max((edu_levels.get(level, 0) for level in edu_levels if level in cv_edu_lower), default=0)
    jd_level = max((edu_levels.get(level, 0) for level in edu_levels if level in jd_edu_lower), default=0)
    
    # Calculate base score from fuzzy matching
    base_score = fuzz.token_set_ratio(cv_edu, jd_edu) / 100
    
    # Adjust score based on education level comparison
    if cv_level >= jd_level:
        level_bonus = 0.2  # Bonus for meeting or exceeding required education
    else:
        level_penalty = (jd_level - cv_level) / 4  # Penalty proportional to the difference
        base_score *= (1 - level_penalty)
    
    return min(1.0, base_score + (0.2 if cv_level >= jd_level else 0))

def title_match(cv_title: str, jd_title: str) -> float:
    return SequenceMatcher(None, cv_title.lower(), jd_title.lower()).ratio()

def tfidf_similarity(cv_text: str, jd_text: str) -> float:
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([cv_text, jd_text])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def semantic_similarity(cv_text: str, jd_text: str) -> float:
    cv_emb = model.encode(cv_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    return util.cos_sim(cv_emb, jd_emb).item()

def location_match(cv_text: str, jd_text: str) -> int:
    cv_locations = ["Sofia", "Haskovo", "Bulgaria"]
    jd_locations = ["Sofia", "Bulgaria"]
    return int(any(loc in cv_text and loc in jd_text for loc in cv_locations))

def extract_education(text):
    edu_keywords = ["Bachelor", "Master", "PhD", "BSc", "MSc", "Doctorate", "degree", "diploma"]
    for line in text.split('\n'):
        for kw in edu_keywords:
            if kw.lower() in line.lower():
                return line
    return ""

def extract_title(text):
    match = re.search(r'(Job Title|Position|Role)[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
    return match.group(2).strip() if match else ""

def match_report(cv_data: dict, jd_data: dict) -> dict:
    report = {}
    report['Skill Match'] = skill_match(cv_data['skills'], jd_data['skills'])
    report['Experience Match'] = experience_match(cv_data['text'], jd_data['text'])
    report['Education Match'] = education_match(cv_data['education'], jd_data['education'])
    report['Title Match'] = title_match(cv_data['title'], jd_data['title'])
    report['TF-IDF Similarity'] = tfidf_similarity(cv_data['text'], jd_data['text'])
    report['Semantic Similarity'] = semantic_similarity(cv_data['text'], jd_data['text'])
    report['Location Match'] = location_match(cv_data['text'], jd_data['text'])
    report['Final Score'] = final_match_score(cv_data, jd_data)
    return report

def calculate_skill_importance(skills: set, jd_text: str) -> dict:
    """Calculate importance weights for different skills based on job description context."""
    skill_weights = {}
    jd_text_lower = jd_text.lower()
    
    for skill in skills:
        weight = 1.0
        skill_lower = skill.lower()
        
        # Increase weight for skills mentioned multiple times
        mentions = jd_text_lower.count(skill_lower)
        if mentions > 1:
            weight += min(0.5, mentions * 0.1)  # Cap at 50% boost
        
        # Increase weight for skills mentioned in requirements
        if "required" in jd_text_lower and skill_lower in jd_text_lower[jd_text_lower.find("required"):]:
            weight += 0.3
        
        # Increase weight for skills mentioned early in the JD
        if skill_lower in jd_text_lower[:len(jd_text_lower)//3]:
            weight += 0.2
            
        skill_weights[skill] = weight
    
    # Normalize weights
    total_weight = sum(skill_weights.values())
    if total_weight > 0:
        skill_weights = {k: v/total_weight for k, v in skill_weights.items()}
    
    return skill_weights

# --- Lightweight domain classifier ---
def detect_domain(text: str) -> str:
    """
    Infer general job domain from text using TF-IDF centroid similarity.
    Returns one of: HR, IT, Finance, Logistics, Marketing, Other.
    """
    domains = {
        "HR": ["human resources", "recruitment", "employee relations", "onboarding", "talent acquisition"],
        "IT": ["developer", "software", "data", "engineer", "programming", "python", "sap", "system"],
        "Finance": ["accounting", "audit", "financial", "budget", "tax", "investment"],
        "Logistics": ["supply chain", "warehouse", "transportation", "logistics", "shipment", "inventory"],
        "Marketing": ["marketing", "branding", "campaign", "seo", "advertising", "digital media"]
    }

    # Prepare corpus (one doc per domain + target text)
    corpus = list(domains.values())
    corpus = [" ".join(words) for words in corpus] + [text.lower()]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    domain_labels = list(domains.keys())
    best_idx = int(np.argmax(sims))
    best_score = sims[best_idx]

    # Apply confidence threshold
    return domain_labels[best_idx] if best_score > 0.15 else "Other"

from sentence_transformers import SentenceTransformer, util

# --- Pretrained semantic domain classifier ---
_model = SentenceTransformer("all-MiniLM-L6-v2")

# Predefined domain prototypes (can expand)
_DOMAIN_PROTOTYPES = {
    "HR": "human resources recruitment employee relations onboarding training payroll",
    "IT": "software development programming data engineer system design cloud devops sap",
    "Finance": "accounting auditing financial analysis budgeting tax banking investment",
    "Logistics": "supply chain transportation warehouse logistics shipping inventory sap ewm",
    "Marketing": "marketing branding campaign seo advertising digital media content",
    "Healthcare": "medical healthcare patient clinical hospital nursing pharmaceutical"
}

def detect_domain(text: str) -> str:
    """
    Determine text's domain using semantic embeddings.
    Returns the domain with highest cosine similarity, else 'Other'.
    """
    if not text or len(text.strip()) < 50:
        return "Other"

    text_emb = _model.encode(text.lower(), convert_to_tensor=True)
    domain_labels, domain_texts = zip(*_DOMAIN_PROTOTYPES.items())
    domain_embs = _model.encode(domain_texts, convert_to_tensor=True)

    sims = util.cos_sim(text_emb, domain_embs)[0]
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])

    return domain_labels[best_idx] if best_score > 0.25 else "Other"

def final_match_score(cv_data: dict, jd_data: dict) -> float:
    """
    Compute a more discriminative final score between CV and Job Description.
    Penalizes domain divergence and generic overlap for higher accuracy.
    """
    # --- Extract component scores ---
    skill = skill_match(cv_data['skills'], jd_data['skills'])
    sem = semantic_similarity(cv_data['text'], jd_data['text'])
    tfidf = tfidf_similarity(cv_data['text'], jd_data['text'])
    edu = education_match(cv_data['education'], jd_data['education'])
    exp = experience_match(cv_data['text'], jd_data['text'])
    title = title_match(cv_data['title'], jd_data['title'])
    loc = location_match(cv_data['text'], jd_data['text'])

    jd_text = jd_data.get("text", "").lower()
    cv_text = cv_data.get("text", "").lower()

    # --- Automatic domain classification ---
    cv_domain = detect_domain(cv_data.get("text", ""))
    jd_domain = detect_domain(jd_data.get("text", ""))

    # --- Domain divergence penalty ---
    domain_penalty = 1.0
    if cv_domain != jd_domain and "Other" not in (cv_domain, jd_domain):
        domain_penalty = 0.6  # up to 40% reduction for cross-domain mismatch


    # --- Correlation normalization ---
    correction_factor = 1 - abs(title - sem) * 0.1
    sem *= correction_factor
    title *= correction_factor

    # --- TF-IDF & Semantic calibration ---
    # If semantic is high but skills low, reduce semantic (false positive)
    if sem > 0.7 and skill < 0.3:
        sem *= 0.7
    # If tfidf and sem disagree strongly, balance them
    if abs(tfidf - sem) > 0.3:
        sem = (sem + tfidf) / 2

    # --- Weight distribution ---
    weights = {
        "skills": 0.15,
        "semantic": 0.35,
        "tfidf": 0.20,
        "education": 0.10,
        "experience": 0.10,
        "title": 0.05,
        "location": 0.05
    }

    base = (
        weights["skills"] * skill +
        weights["semantic"] * sem +
        weights["tfidf"] * tfidf +
        weights["education"] * edu +
        weights["experience"] * exp +
        weights["title"] * title +
        weights["location"] * loc
    )

    # --- Domain penalty applied globally ---
    base *= domain_penalty

    # --- Consistency bonus ---
    consistency = np.mean([skill, sem, tfidf])
    if consistency > 0.7:
        base += 0.05 * consistency

    # --- Soft penalty for fundamentals ---
    if skill < 0.4 or sem < 0.4:
        base *= 0.9

    # --- Calibration ---
    base = min(1.0, base + 0.05)
    final = 1 / (1 + np.exp(-4.5 * (base - 0.45)))
    final = float(np.clip(final, 0.0, 1.0))

    return round(final, 3)

    # --- Interpretation tiers ---
    # 0.00–0.39 → Weak Fit
    # 0.40–0.59 → Partial Fit (trainable)
    # 0.60–0.74 → Good Fit
    # 0.75–1.00 → Excellent Fit

# Prepare data for scoring
edu_cv = extract_education(cv_text)
edu_jd = extract_education(jd_text)
title_cv = extract_title(cv_text)
title_jd = extract_title(jd_text)

cv_data = {
    'skills': cv_skills,
    'text': cv_text_clean,
    'title': title_cv,
    'education': edu_cv,
}

jd_data = {
    'skills': jd_skills,
    'text': jd_text_clean,
    'title': title_jd,
    'education': edu_jd,
}

report = match_report(cv_data, jd_data)
for k, v in report.items():
    if isinstance(v, float):
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v}")

# --- Visualization ---
import matplotlib.pyplot as plt
import numpy as np
import glob


def visualize_report(report):
    categories = list(report.keys())[:-1]  # exclude final score
    scores = [report[k] if isinstance(report[k], float) else float(report[k]) for k in categories]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(categories, scores, color='skyblue')
    ax.set_ylim(0, 1)
    ax.set_title("CV vs Job Description Match Report (Bar Chart)")
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel("Score")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')
    fig.tight_layout()
    return fig

def visualize_radar(report):
    categories = list(report.keys())[:-1]  # exclude final score
    scores = [report[k] if isinstance(report[k], float) else float(report[k]) for k in categories]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, 'o-', linewidth=2, label='Match Scores')
    ax.fill(angles, scores, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1)
    ax.set_title("CV vs JD Match Report (Radar Chart)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    fig.tight_layout()
    return fig

# Batch ranking function
def rank_cvs(jd_data, cv_folder):
    ranking = []
    for cv_path in glob.glob(os.path.join(cv_folder, "*.pdf")):
        cv_text = extract_text_from_pdf(cv_path)
        cv_text_clean = re.sub(r'\s+', ' ', cv_text)
        cv_data = {
            'text': cv_text_clean,
            'skills': extract_skills(cv_text_clean),
            'title': extract_title(cv_text_clean),
            'education': extract_education(cv_text_clean)
        }
        score = final_match_score(cv_data, jd_data)
        report = match_report(cv_data, jd_data)
        ranking.append((cv_path, score, report))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking

# Example usage for batch ranking and radar chart
# Uncomment and set your folder and JD file to use
# jd_file = "sample-job-description.pdf"
# cv_folder = "cvs_folder"
# jd_text = extract_text_from_pdf(jd_file)
# jd_text_clean = re.sub(r'\s+', ' ', jd_text)
# jd_data = {
#     'text': jd_text_clean,
#     'skills': extract_skills(jd_text_clean),
#     'title': extract_title(jd_text_clean),
#     'education': extract_education(jd_text_clean)
# }
# ranking = rank_cvs(jd_data, cv_folder)
# for i, (cv_file, score, report) in enumerate(ranking[:5]):
#     print(f"{i+1}. {cv_file} - Score: {score:.2f}")
#     visualize_radar(report)

visualize_report(report)
visualize_radar(report)
