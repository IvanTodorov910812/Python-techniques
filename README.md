#Run the application via codespaces  
python3 -m pip install -r requirements.txt  
streamlit run dashboard.py

Frontend uploads a PDF → Edge Function passes a base64 payload to Python → extract_text→ clean_text → return JSON with text.

🚀 Summary
| Area                      | Current                     | Proposed Improvement                                       |
| ------------------------- | --------------------------- | ---------------------------------------------------------- |
| **Final Score Algorithm** | Overboosted, uneven weights | Balanced, sigmoid-scaled composite model                 |
| **ESCO Taxonomy**         | ESCO Classification | Standardized across 27+ countries, up to 85% match precision
| **Skill Matching**        | Flat Jaccard + fuzzy        | Category-weighted with contextual importance               |
| **Experience Evaluation** | Regex year check            | NLP-based timeline & duration consistency                  |
| **Education Handling**    | Fuzzy + level bonus         | Hierarchical weighting by degree field relevance           |
| **Visualization**         | Plotly, Radar, WordCloud    | Competency Heatmap                            |
| **Red Flags**             | Heuristic                   | ML-based anomaly detection                             |
| **HR Functions**          | Match & Rank                | Add bias detection, feedback loop, and shortlisting export | 

🔍 Candidate Scoring Enhancements

Weighted Skill Category Scoring — Split skills into technical, soft, and domain categories with separate match percentages.

Experience Timeline Validation — Detect gaps or overlapping roles using NLP + regex temporal extraction.

Certifications Match — Recognize relevant certifications (AWS, PMP, etc.) and weigh them like micro-qualifications.

Role-Level Calibration — Adjust scoring logic depending on job seniority (e.g., “junior” roles emphasize learning potential).

🧭 HR Analytics & Diversity Insights

Bias Detection Module — Check for gendered wording or biased role phrasing in JD or CV (inclusive hiring compliance).

Diversity Score — Incorporate anonymized background variety as a non-weighted data insight (for HR reporting).

📊 Recruiter Workflow Tools

Candidate Comparison Dashboard — Strength/weakness radar heatmaps per role competency.

Automated Shortlist Report — Export top matches with gap-analysis and suggested interview topics.

💬 Interview Preparation Tools

Behavioral Question Generator 2.0 — Use candidate weak areas to generate STAR-based interview prompts (“Tell me about a time…”).

Cultural Fit Predictor — Analyze tone and phrasing in CV summary to match company values or leadership principles.

⚙️ Data Quality & Compliance

GDPR Compliance Checks — Flag personal identifiers in CVs (address, birth date, etc.) for anonymization.

Document Quality Scoring — Penalize low-quality scans or missing structured data (incomplete PDFs).
