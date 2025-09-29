
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
    intersection = cv_skills & jd_skills
    union = cv_skills | jd_skills
    return len(intersection) / len(union)

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
    return fuzz.token_set_ratio(cv_edu, jd_edu) / 100

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

def final_match_score(cv_data: dict, jd_data: dict) -> float:
    score = 0
    score += 0.3 * skill_match(cv_data['skills'], jd_data['skills'])
    score += 0.2 * tfidf_similarity(cv_data['text'], jd_data['text'])
    score += 0.2 * semantic_similarity(cv_data['text'], jd_data['text'])
    score += 0.1 * title_match(cv_data['title'], jd_data['title'])
    score += 0.1 * education_match(cv_data['education'], jd_data['education'])
    score += 0.05 * experience_match(cv_data['text'], jd_data['text'])
    score += 0.05 * location_match(cv_data['text'], jd_data['text'])
    return score

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
