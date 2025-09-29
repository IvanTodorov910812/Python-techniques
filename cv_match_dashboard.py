import streamlit as st
import tempfile
import os
from extract_from_PDF import (
    extract_text_from_pdf,
    extract_skills,
    extract_title,
    extract_education,
    match_report,
    visualize_report,
    visualize_radar,
    rank_cvs
)

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

    cv_data = {
        'text': cv_text,
        'skills': extract_skills(cv_text),
        'title': extract_title(cv_text),
        'education': extract_education(cv_text)
    }
    jd_data = {
        'text': jd_text,
        'skills': extract_skills(jd_text),
        'title': extract_title(jd_text),
        'education': extract_education(jd_text)
    }

    report = match_report(cv_data, jd_data)

    st.write(report)

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



    st.markdown("### Visualization (Bar Chart)")
    fig_bar = visualize_report(report)
    st.pyplot(fig_bar)
    bar_pdf = None
    import io
    bar_buf = io.BytesIO()
    fig_bar.savefig(bar_buf, format="pdf")
    bar_pdf = bar_buf.getvalue()
    if st.button("Export Bar Chart as PDF"):
        st.download_button(
            label="Download Bar Chart PDF",
            data=bar_pdf,
            file_name="bar_chart.pdf",
            mime="application/pdf"
        )

    st.markdown("### Visualization (Radar Chart)")
    fig_radar = visualize_radar(report)
    st.pyplot(fig_radar)
    radar_pdf = None
    radar_buf = io.BytesIO()
    fig_radar.savefig(radar_buf, format="pdf")
    radar_pdf = radar_buf.getvalue()
    if st.button("Export Radar Chart as PDF"):
        st.download_button(
            label="Download Radar Chart PDF",
            data=radar_pdf,
            file_name="radar_chart.pdf",
            mime="application/pdf"
        )

    # Batch ranking UI
    st.markdown("### Batch Ranking of CVs")
    cv_folder = st.text_input("Enter folder path containing CV PDFs for batch ranking:")
    if cv_folder:
        ranking = rank_cvs(jd_data, cv_folder)
        for i, (cv_file_path, score, cv_report) in enumerate(ranking[:5]):
            st.write(f"{i+1}. {os.path.basename(cv_file_path)} - Score: {score:.2f}")
            st.write(cv_report)
            fig_radar = visualize_radar(cv_report)
            st.pyplot(fig_radar)

    # Placeholder for LLM explanation
    st.markdown("### Explanation")
    st.write("LLM-powered explanation coming soon!")

    # Clean up temp files
    os.remove(cv_path)
    os.remove(jd_path)
