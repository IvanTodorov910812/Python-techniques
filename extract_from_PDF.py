
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
