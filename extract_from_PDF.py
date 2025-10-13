
import fitz  # PyMuPDF
import pdfplumber
import layoutparser as lp
import os

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
print("0.40 – 0.59\tWeak match, needs review")

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

def final_match_score(cv_data: dict, jd_data: dict) -> float:
    # Calculate individual scores
    skill_score = skill_match(cv_data['skills'], jd_data['skills'])
    tfidf_score = tfidf_similarity(cv_data['text'], jd_data['text'])
    semantic_score = semantic_similarity(cv_data['text'], jd_data['text'])
    title_score = title_match(cv_data['title'], jd_data['title'])
    education_score = education_match(cv_data['education'], jd_data['education'])
    experience_score = experience_match(cv_data['text'], jd_data['text'])
    location_score = location_match(cv_data['text'], jd_data['text'])
    
    # Calculate skill importance weights
    skill_weights = calculate_skill_importance(jd_data['skills'], jd_data['text'])
    
    # Enhanced scoring system with dynamic weights
    # Primary criteria (must-haves) - 75% of base score
    primary_weights = {
        'skills': 0.45,     # Most critical
        'semantic': 0.30,   # Overall relevance
        'qual': 0.25       # Qualifications (education/experience)
    }
    
    # Calculate qualification score with progressive bonuses
    qual_score = max(experience_score, education_score)  # Base qualification
    if experience_score > 0.7 and education_score > 0.7:
        qual_score = max(qual_score, (experience_score + education_score) / 1.8)  # Bonus for both high
    
    primary_score = (
        primary_weights['skills'] * skill_score +
        primary_weights['semantic'] * semantic_score +
        primary_weights['qual'] * qual_score
    )
    
    # Secondary criteria (nice-to-haves) - 25% of base score
    secondary_weights = {
        'tfidf': 0.35,      # Keyword relevance
        'title': 0.30,      # Role alignment
        'extra_qual': 0.25, # Additional qualifications
        'location': 0.10    # Location match
    }
    
    secondary_score = (
        secondary_weights['tfidf'] * tfidf_score +
        secondary_weights['title'] * title_score +
        secondary_weights['extra_qual'] * min(experience_score, education_score) +
        secondary_weights['location'] * location_score
    )
    
    # Calculate base score with dynamic weighting
    base_score = (0.75 * primary_score + 0.25 * secondary_score)
    
    # Enhanced boosting system
    boosters = 1.0
    
    # Progressive skill match boosting
    if skill_score > 0.9:
        boosters += 0.25  # Exceptional skill match
    elif skill_score > 0.8:
        boosters += 0.15  # Strong skill match
    elif skill_score > 0.7:
        boosters += 0.10  # Good skill match
    
    # Qualification excellence boost
    if experience_score > 0.8 and education_score > 0.8:
        boosters += 0.15
    elif experience_score > 0.7 and education_score > 0.7:
        boosters += 0.10
    
    # Semantic relevance boost
    if semantic_score > 0.8:
        boosters += 0.15
    elif semantic_score > 0.7:
        boosters += 0.10
    
    # Title alignment boost
    if title_score > 0.9:
        boosters += 0.10
    
    # Comprehensive excellence boost
    if all(score > 0.7 for score in [skill_score, semantic_score, qual_score]):
        boosters += 0.15
    
    # Progressive penalty system
    penalties = 1.0
    
    # Critical skills penalty
    if skill_score < 0.3:
        penalties -= 0.30
    elif skill_score < 0.4:
        penalties -= 0.20
    elif skill_score < 0.5:
        penalties -= 0.10
    
    # Semantic relevance penalty
    if semantic_score < 0.3:
        penalties -= 0.25
    elif semantic_score < 0.4:
        penalties -= 0.15
    
    # Qualification penalty
    if qual_score < 0.3:
        penalties -= 0.20
    
    # Calculate final score
    final_score = base_score * boosters * penalties
    
    # Normalize score
    final_score = min(1.0, max(0.0, final_score))
    
    # Apply enhanced distribution curve
    if final_score > 0.3:
        # Custom sigmoid that maintains more granularity in mid-range
        # while still providing good separation at the extremes
        x = 2.5 * (final_score - 0.5)
        final_score = 0.5 + 0.5 * (x / np.sqrt(1 + x*x))
    
    return final_score

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
