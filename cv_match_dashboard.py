import streamlit as st
import tempfile
import os
from extract_from_PDF import (
    extract_text_from_pdf,
    extract_skills,
    extract_title,
    extract_education,
    match_report,
    visualize_report
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

    st.markdown("### Visualization")
    st.pyplot(visualize_report(report))

    # Placeholder for LLM explanation
    st.markdown("### Explanation")
    st.write("LLM-powered explanation coming soon!")

    # Clean up temp files
    os.remove(cv_path)
    os.remove(jd_path)
