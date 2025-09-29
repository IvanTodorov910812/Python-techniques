import streamlit as st
import tempfile
import os
import io
import base64
import re
import plotly.graph_objs as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime
import pandas as pd
from extract_from_PDF import (
    extract_text_from_pdf,
    extract_skills,
    extract_title,
    extract_education,
    match_report,
    rank_cvs
)

# --- Skill Taxonomy Normalization ---
# --- Enhanced Visualization Functions ---
def create_quadrant_chart(cv_data, jd_data):
    skills = list(cv_data['skills'] | jd_data['skills'])
    x_values = []  # CV proficiency
    y_values = []  # JD importance
    labels = []
    
    for skill in skills:
        x_values.append(1.0 if skill in cv_data['skills'] else 0.0)
        y_values.append(1.0 if skill in jd_data['skills'] else 0.0)
        labels.append(skill)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers+text',
        text=labels,
        textposition="top center",
        name='Skills'
    ))
    
    # Add quadrant lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    
    # Update layout with quadrant labels
    fig.update_layout(
        title="Skills Quadrant Analysis",
        xaxis_title="CV Proficiency",
        yaxis_title="JD Importance",
        annotations=[
            dict(x=0.25, y=0.75, text="Development Needed", showarrow=False),
            dict(x=0.75, y=0.75, text="Strengths", showarrow=False),
            dict(x=0.25, y=0.25, text="Low Priority", showarrow=False),
            dict(x=0.75, y=0.25, text="Potential Overqualification", showarrow=False)
        ]
    )
    return fig

def create_timeline_comparison(cv_text, jd_text):
    # Extract years from CV and JD
    cv_years = sorted(list(map(int, re.findall(r'20\d{2}', cv_text))))
    jd_years = sorted(list(map(int, re.findall(r'20\d{2}', jd_text))))
    
    if not cv_years or not jd_years:
        return None
        
    fig = go.Figure()
    
    # Add CV timeline
    fig.add_trace(go.Scatter(
        x=cv_years,
        y=[1] * len(cv_years),
        mode='markers+lines',
        name='CV Timeline',
        line=dict(color='blue')
    ))
    
    # Add JD requirements timeline
    fig.add_trace(go.Scatter(
        x=jd_years,
        y=[0] * len(jd_years),
        mode='markers+lines',
        name='JD Requirements',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="Timeline Comparison",
        yaxis=dict(
            ticktext=["JD Timeline", "CV Timeline"],
            tickvals=[0, 1],
            range=[-0.5, 1.5]
        ),
        xaxis_title="Year"
    )
    return fig

def create_word_cloud(cv_text, jd_text):
    # Create word frequency dictionaries
    cv_words = Counter(cv_text.lower().split())
    jd_words = Counter(jd_text.lower().split())
    
    # Create comparison word cloud
    combined_words = {}
    for word in set(cv_words.keys()) | set(jd_words.keys()):
        cv_freq = cv_words.get(word, 0)
        jd_freq = jd_words.get(word, 0)
        if cv_freq > 0 and jd_freq > 0:
            combined_words[word] = cv_freq + jd_freq
    
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud.generate_from_frequencies(combined_words)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def create_funnel_chart(cv_scores, criteria_labels):
    fig = go.Figure(go.Funnel(
        y=criteria_labels,
        x=cv_scores,
        textinfo="value+percent initial"
    ))
    
    fig.update_layout(
        title="CV Evaluation Funnel",
        showlegend=False
    )
    return fig

def create_comparison_matrix(candidates):
    # Create DataFrame for comparison
    df = pd.DataFrame(candidates)
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    
    fig.update_layout(
        title="Candidate Comparison Matrix",
        height=400
    )
    return fig

def normalize_skills(skills):
    # Example normalization: lowercase, strip, map synonyms
    taxonomy = {
        'python': ['python', 'python3', 'py'],
        'machine learning': ['machine learning', 'ml'],
        'data analysis': ['data analysis', 'data analytics', 'analytics'],
        'sql': ['sql', 'mysql', 'postgresql', 'sqlite'],
        'excel': ['excel', 'ms excel', 'microsoft excel'],
        # Add more mappings as needed
    }
    norm = set()
    for skill in skills:
        s = skill.lower().strip()
        found = False
        for key, vals in taxonomy.items():
            if s in vals:
                norm.add(key)
                found = True
                break
        if not found:
            norm.add(s)
    return norm

# --- Red Flag Detection ---
def detect_red_flags(cv_text, jd_text):
    import re
    flags = []
    cv_text_lower = cv_text.lower()
    jd_text_lower = jd_text.lower()

    # 1. Missing required skills (under-qualification)
    required_keywords = ['python', 'sql', 'machine learning']
    for req in required_keywords:
        if req in jd_text_lower and req not in cv_text_lower:
            flags.append(f"Missing required skill: {req}")

    # 2. Career hopping (multiple short tenures < 1 year)
    # Look for patterns like: Company, 2021-2022, 2022-2023, etc.
    years = re.findall(r'(20\d{2})', cv_text)
    if years:
        years = sorted(set(map(int, years)))
        short_tenures = 0
        for i in range(1, len(years)):
            if years[i] - years[i-1] <= 1:
                short_tenures += 1
        if short_tenures >= 2:
            flags.append("Possible career hopping: multiple short tenures detected")

    # 3. Over-qualification (PhD applying for entry-level)
    if ('phd' in cv_text_lower or 'doctorate' in cv_text_lower) and ('entry level' in jd_text_lower or 'junior' in jd_text_lower):
        flags.append("Possible over-qualification: PhD/Doctorate for entry-level role")

    # 4. Under-qualification (missing critical requirements)
    # Already partly covered by missing required skills
    if 'bachelor' in jd_text_lower and 'bachelor' not in cv_text_lower and 'master' not in cv_text_lower and 'phd' not in cv_text_lower:
        flags.append("Missing required degree: Bachelor or higher")

    # 5. Location mismatch with no relocation mention
    # Look for city/country in JD but not in CV, and no 'relocation' in CV
    locations = ['sofia', 'bulgaria', 'london', 'remote']
    for loc in locations:
        if loc in jd_text_lower and loc not in cv_text_lower and 'relocat' not in cv_text_lower:
            flags.append(f"Location mismatch: {loc.title()} required, not found in CV and no relocation mentioned")

    # 6. Overused buzzwords
    buzzwords = ['synergy', 'rockstar', 'ninja', 'guru', 'thought leader', 'disrupt', 'pivot', 'game changer']
    for buzz in buzzwords:
        if buzz in cv_text_lower:
            flags.append(f"Overused buzzword detected: '{buzz}'")

    # 7. Inconsistencies (date overlaps, role progression logic)
    # Look for overlapping years (e.g., two jobs with same year)
    year_counts = {}
    for y in re.findall(r'(20\d{2})', cv_text):
        year_counts[y] = year_counts.get(y, 0) + 1
    overlaps = [y for y, c in year_counts.items() if c > 1]
    if overlaps:
        flags.append(f"Possible date overlap(s) in CV: {', '.join(overlaps)}")

    # Simple role progression: if 'manager' before 'junior' in CV
    manager_idx = cv_text_lower.find('manager')
    junior_idx = cv_text_lower.find('junior')
    if manager_idx != -1 and junior_idx != -1 and manager_idx < junior_idx:
        flags.append("Role progression inconsistency: 'Manager' listed before 'Junior'")

    # Example: employment gap (very basic, keep as last)
    if '2019' in cv_text and '2021' not in cv_text:
        flags.append("Possible employment gap after 2019")

    return flags

st.title("CV vs Job Description Matcher")

cv_file = st.file_uploader("Upload CV PDF", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"])


if cv_file and jd_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_cv:
        tmp_cv.write(cv_file.read())
        cv_path = tmp_cv.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_jd:
        tmp_jd.write(jd_file.read())
        jd_path = tmp_jd.name

    cv_text = extract_text_from_pdf(cv_path)
    jd_text = extract_text_from_pdf(jd_path)

    # Skill extraction and normalization
    cv_skills_raw = extract_skills(cv_text)
    jd_skills_raw = extract_skills(jd_text)
    cv_skills = normalize_skills(cv_skills_raw)
    jd_skills = normalize_skills(jd_skills_raw)
    cv_data = {
        'text': cv_text,
        'skills': cv_skills,
        'title': extract_title(cv_text),
        'education': extract_education(cv_text)
    }
    jd_data = {
        'text': jd_text,
        'skills': jd_skills,
        'title': extract_title(jd_text),
        'education': extract_education(jd_text)
    }


    report = match_report(cv_data, jd_data)

    # Red flag detection
    red_flags = detect_red_flags(cv_text, jd_text)

    st.write(report)
    if red_flags:
        st.error("Red Flags Detected:")
        for flag in red_flags:
            st.write(f"- {flag}")

    st.markdown("---")
    st.markdown("#### Match Feature Descriptions")
    st.markdown("""
* **Skill Match**: Measures the overlap of skills/keywords between the CV and Job Description using Jaccard similarity.
* **Experience Match**: Checks if the candidate's years of experience meet or exceed the requirement in the Job Description.
* **Education Match**: Compares the education level and field using fuzzy string matching.
* **Title Match**: Assesses how closely the candidate's job titles match the target role/title in the Job Description.
* **TF-IDF Similarity**: Evaluates overall text similarity using term frequency-inverse document frequency and cosine similarity.
* **Semantic Similarity**: Uses sentence-transformers to measure deep semantic similarity between the CV and Job Description.
* **Location Match**: Checks if the candidate's location matches the job location (simple substring match).
* **Final Score**: Weighted combination of all features, representing the overall match between CV and Job Description.
    """)



    # --- Enhanced Visualization with Plotly ---
    st.markdown("### Visualization (Bar Chart)")
    categories = list(report.keys())[:-1]
    scores = [report[k] if isinstance(report[k], float) else float(report[k]) for k in categories]
    bar_fig = go.Figure([go.Bar(x=categories, y=scores, marker_color='skyblue')])
    bar_fig.update_layout(title="CV vs Job Description Match Report (Bar Chart)", yaxis=dict(range=[0,1]))
    st.plotly_chart(bar_fig, use_container_width=True, key="bar_chart")

    # --- Interview Question Generator ---
    def generate_interview_questions(cv_data, jd_data, red_flags, report):
        all_questions = {
            'technical': [],
            'behavioral': [],
            'match_specific': {
                'Skill Match': [],
                'Experience Match': [],
                'Education Match': [],
                'Title Match': [],
                'Location Match': []
            }
        }
        
        # 1. Technical Questions
        for skill in jd_data.get('skills', []):
            if skill not in cv_data.get('skills', []):
                all_questions['technical'].append(f"We may include a technical assessment on {skill}. How comfortable are you with this topic?")
            else:
                all_questions['technical'].append(f"Can you describe a project where you used {skill}?")
        
        # 2. Behavioral Questions
        for skill in list(cv_data.get('skills', []))[:5]:
            all_questions['behavioral'].append(f"Tell me about a time you used {skill} in a project or work setting.")
        
        # 3. Match-Specific Questions
        # Skill Match Questions
        skill_gaps = jd_data.get('skills', set()) - cv_data.get('skills', set())
        for skill in list(skill_gaps)[:3]:
            all_questions['match_specific']['Skill Match'].append(f"How do you plan to acquire expertise in {skill}?")
        all_questions['match_specific']['Skill Match'].extend([
            "Which of your technical skills do you believe are most relevant for this role?",
            "How do you stay updated with new technologies in your field?"
        ])
        
        # Experience Match Questions
        all_questions['match_specific']['Experience Match'].extend([
            "What aspects of your previous roles align most closely with this position?",
            "Can you describe a challenging project that demonstrates your relevant experience?",
            "How has your experience prepared you for the responsibilities of this role?",
            "What unique perspective would you bring based on your past experience?",
            "How do you apply lessons from your previous roles to new challenges?"
        ])
        
        # Education Match Questions
        all_questions['match_specific']['Education Match'].extend([
            "How has your educational background prepared you for this role?",
            "What relevant coursework or projects have you completed?",
            "How do you continue your professional development?",
            "What certifications or additional training do you plan to pursue?",
            "How do you apply your academic knowledge in practical situations?"
        ])
        
        # Title Match Questions
        all_questions['match_specific']['Title Match'].extend([
            "How does this role align with your career progression?",
            "What attracted you to this specific position?",
            "Where do you see yourself professionally in the next few years?",
            "How would this role help you achieve your career goals?",
            "What aspects of this position most interest you?"
        ])
        
        # Location Match Questions
        all_questions['match_specific']['Location Match'].extend([
            "What is your preferred work arrangement (remote/hybrid/onsite)?",
            "Are you willing to relocate or travel for this position?",
            "How would you handle time zone differences if working remotely?",
            "What experience do you have with virtual collaboration?",
            "How do you maintain work-life balance in different work arrangements?"
        ])
        
        # Add red flag related questions to appropriate categories
        for flag in red_flags:
            if "Missing required skill" in flag:
                skill = flag.split(":")[-1].strip()
                all_questions['technical'].append(f"Can you elaborate on your experience or willingness to learn {skill}?")
            if "employment gap" in flag or "career hopping" in flag:
                all_questions['behavioral'].append("Can you explain your career progression and any gaps?")
            if "role progression" in flag:
                all_questions['match_specific']['Experience Match'].append("Can you clarify your role progression and responsibilities?")
        
        # Trim all question lists to maximum 5 questions each
        for category in ['technical', 'behavioral']:
            all_questions[category] = all_questions[category][:5]
        for match_type in all_questions['match_specific']:
            all_questions['match_specific'][match_type] = all_questions['match_specific'][match_type][:5]
        
        return all_questions

    # --- Display Interview Question Generator ---
    st.markdown("### Interview Question Generator")
    questions = generate_interview_questions(cv_data, jd_data, red_flags, report)
    if questions:
        # Technical and Behavioral Questions in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('**🔧 Technical Questions**')
            for i, q in enumerate(questions['technical'], 1):
                st.write(f'{i}. {q}')
                
        with col2:
            st.markdown('**� Behavioral Questions**')
            for i, q in enumerate(questions['behavioral'], 1):
                st.write(f'{i}. {q}')
        
        # Match-specific questions in expandable sections
        st.markdown('\n### Match-Specific Questions')
        for match_type, match_questions in questions['match_specific'].items():
            if match_questions:
                with st.expander(f"� {match_type} Questions"):
                    for i, q in enumerate(match_questions, 1):
                        st.write(f'{i}. {q}')
    else:
        st.write("No specific interview questions generated.")

    st.markdown("### Visualization (Radar Chart)")
    radar_scores = scores + scores[:1]
    radar_categories = categories + [categories[0]]
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=radar_scores, theta=radar_categories, fill='toself', name='Match Scores'))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=False,
        title="CV vs JD Match Report (Radar Chart)"
    )
    st.plotly_chart(radar_fig, use_container_width=True, key="radar_chart")

    # Create funnel data
    funnel_scores = [
        len(cv_data['skills']),  # Total skills
        len(cv_data['skills'] & jd_data['skills']),  # Matching skills
        sum([1 for flag in red_flags if "Missing required skill" not in flag]),  # Passing criteria
        report['Final Score'] * 100  # Final score as percentage
    ]
    funnel_labels = [
        "Total Skills",
        "Matching Skills",
        "Passing Criteria",
        "Final Score"
    ]

    st.markdown("### Evaluation Funnel")
    funnel_fig = create_funnel_chart(funnel_scores, funnel_labels)
    st.plotly_chart(funnel_fig, use_container_width=True, key="funnel_chart")

    # --- Enhanced Visualizations ---
    st.markdown("### Skills Quadrant Analysis")
    quadrant_fig = create_quadrant_chart(cv_data, jd_data)
    st.plotly_chart(quadrant_fig, use_container_width=True, key="quadrant_chart")

    st.markdown("### Career Timeline Comparison")
    timeline_fig = create_timeline_comparison(cv_text, jd_text)
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True, key="timeline_chart")
    else:
        st.info("Not enough timeline data to create comparison.")

    st.markdown("### Word Cloud Analysis")
    wordcloud_fig = create_word_cloud(cv_text, jd_text)
    st.pyplot(wordcloud_fig)

    # --- PDF Export of Full Report ---
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib import colors
    import tempfile as _tempfile
    import matplotlib.pyplot as plt
    import plotly.io as pio

    def plotly_to_image(fig, fmt='png'):
        import base64
        # Convert plot to HTML with static image
        img_bytes = fig.to_image(format=fmt, width=1200, height=800)
        # Convert to base64 for embedding
        img_base64 = base64.b64encode(img_bytes).decode()
        
        # Create an HTML file with the embedded image
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(f'''
            <html>
            <head><title>Chart</title></head>
            <body style="margin:0;padding:0;">
                <img src="data:image/{fmt};base64,{img_base64}" style="width:100%;height:auto;">
            </body>
            </html>
            ''')
        
        # Use wkhtmltoimage to convert to PNG
        output_path = f.name.replace('.html', f'.{fmt}')
        os.system(f'wkhtmltoimage --quiet --width 1200 {f.name} {output_path}')
        
        # Read the output image
        with open(output_path, 'rb') as img_file:
            output_bytes = img_file.read()
        
        # Clean up temporary files
        os.remove(f.name)
        os.remove(output_path)
        
        return output_bytes

    def export_full_report(report, red_flags, bar_fig, radar_fig, quadrant_fig, timeline_fig, wordcloud_fig, funnel_fig):
        with _tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            c = canvas.Canvas(tmp_pdf.name, pagesize=letter)
            width, height = letter
            y = height - 40
            c.setFont("Helvetica-Bold", 18)
            c.drawString(40, y, "CV vs Job Description Match Report")
            y -= 30
            c.setFont("Helvetica", 12)
            c.drawString(40, y, f"Final Score: {report.get('Final Score', 0):.2f}")
            y -= 20
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, y, "Feature Scores:")
            y -= 20
            c.setFont("Helvetica", 11)
            for k, v in report.items():
                if k == 'Final Score':
                    continue
                c.drawString(60, y, f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
                y -= 16
            y -= 10
            if red_flags:
                c.setFont("Helvetica-Bold", 13)
                c.setFillColor(colors.red)
                c.drawString(40, y, "Red Flags Detected:")
                c.setFillColor(colors.black)
                y -= 18
                c.setFont("Helvetica", 11)
                for flag in red_flags:
                    c.drawString(60, y, f"- {flag}")
                    y -= 14
                y -= 10
            # Feature Descriptions
            c.setFont("Helvetica-Bold", 13)
            c.drawString(40, y, "Match Feature Descriptions:")
            y -= 18
            c.setFont("Helvetica", 10)
            desc_lines = [
                "Skill Match: Measures the overlap of skills/keywords between the CV and Job Description using Jaccard similarity.",
                "Experience Match: Checks if the candidate's years of experience meet or exceed the requirement in the Job Description.",
                "Education Match: Compares the education level and field using fuzzy string matching.",
                "Title Match: Assesses how closely the candidate's job titles match the target role/title in the Job Description.",
                "TF-IDF Similarity: Evaluates overall text similarity using term frequency-inverse document frequency and cosine similarity.",
                "Semantic Similarity: Uses sentence-transformers to measure deep semantic similarity between the CV and Job Description.",
                "Location Match: Checks if the candidate's location matches the job location (simple substring match).",
                "Final Score: Weighted combination of all features, representing the overall match between CV and Job Description."
            ]
            for line in desc_lines:
                c.drawString(60, y, f"- {line}")
                y -= 13
            c.showPage()
            # Add Bar Chart
            bar_img = plotly_to_image(bar_fig)
            bar_reader = ImageReader(io.BytesIO(bar_img))
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, height - 40, "Bar Chart: Match Report")
            c.drawImage(bar_reader, 40, height/2-40, width=500, height=200, preserveAspectRatio=True, mask='auto')
            c.showPage()
            # Add Radar Chart
            radar_img = plotly_to_image(radar_fig)
            radar_reader = ImageReader(io.BytesIO(radar_img))
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, height - 40, "Radar Chart: Match Report")
            c.drawImage(radar_reader, 40, height/2-40, width=400, height=300, preserveAspectRatio=True, mask='auto')
            c.showPage()

            # Add Quadrant Chart
            quadrant_img = plotly_to_image(quadrant_fig)
            quadrant_reader = ImageReader(io.BytesIO(quadrant_img))
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, height - 40, "Skills Quadrant Analysis")
            c.drawImage(quadrant_reader, 40, height/2-40, width=500, height=400, preserveAspectRatio=True, mask='auto')
            c.showPage()

            # Add Timeline Comparison
            if timeline_fig:
                timeline_img = plotly_to_image(timeline_fig)
                timeline_reader = ImageReader(io.BytesIO(timeline_img))
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, height - 40, "Career Timeline Comparison")
                c.drawImage(timeline_reader, 40, height/2-40, width=500, height=200, preserveAspectRatio=True, mask='auto')
                c.showPage()

            # Add Word Cloud
            wordcloud_buf = io.BytesIO()
            wordcloud_fig.savefig(wordcloud_buf, format='png', bbox_inches='tight', pad_inches=0)
            wordcloud_buf.seek(0)
            wordcloud_reader = ImageReader(wordcloud_buf)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, height - 40, "Word Cloud Analysis")
            c.drawImage(wordcloud_reader, 40, height/2-40, width=500, height=250, preserveAspectRatio=True, mask='auto')
            c.showPage()

            # Add Funnel Chart
            funnel_img = plotly_to_image(funnel_fig)
            funnel_reader = ImageReader(io.BytesIO(funnel_img))
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, height - 40, "Evaluation Funnel")
            c.drawImage(funnel_reader, 40, height/2-40, width=400, height=300, preserveAspectRatio=True, mask='auto')
            
            c.save()
            return tmp_pdf.name

    if st.button("Export Full Report as PDF"):
        pdf_path = export_full_report(report, red_flags, bar_fig, radar_fig, quadrant_fig, timeline_fig, wordcloud_fig, funnel_fig)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download Full Report PDF",
                data=f.read(),
                file_name="cv_jd_match_report.pdf",
                mime="application/pdf"
            )

    # Batch ranking UI
    st.markdown("### Batch Ranking of CVs")
    cv_folder = st.text_input("Enter folder path containing CV PDFs for batch ranking:")
    if cv_folder:
        ranking = rank_cvs(jd_data, cv_folder)
        
        # Individual CV results
        for i, (cv_file_path, score, cv_report) in enumerate(ranking[:5]):
            st.write(f"{i+1}. {os.path.basename(cv_file_path)} - Score: {score:.2f}")
            st.write(cv_report)
            fig_radar = visualize_radar(cv_report)
            st.pyplot(fig_radar)
        
        # Comparison Matrix
        if ranking:
            st.markdown("### Candidate Comparison Matrix")
            candidates_data = []
            for cv_file_path, score, cv_report in ranking[:5]:
                candidates_data.append({
                    'Candidate': os.path.basename(cv_file_path),
                    'Score': f"{score:.2f}",
                    'Skills Match': f"{cv_report.get('Skill Match', 0):.2f}",
                    'Experience Match': cv_report.get('Experience Match', 0),
                    'Education Match': f"{cv_report.get('Education Match', 0):.2f}"
                })
            
            if candidates_data:
                matrix_fig = create_comparison_matrix(candidates_data)
                st.plotly_chart(matrix_fig, use_container_width=True, key="comparison_matrix")

    # Placeholder for LLM explanation
    st.markdown("### Explanation")
    st.write("LLM-powered explanation coming soon!")

    # Clean up temp files
    os.remove(cv_path)
    os.remove(jd_path)
