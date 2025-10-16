import streamlit as st
import pickle
import docx
import PyPDF2
import re

# ---------------- Load models ----------------
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# ---------------- Optional NER (spaCy) ----------------
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception as e:
    SPACY_AVAILABLE = False
    print("spaCy not available or model not downloaded:", e)

# ---------------- Helper Functions ----------------

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText.strip()

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return ''.join([page.extract_text() for page in pdf_reader.pages])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type!")

def pred(input_resume):
    cleaned = cleanResume(input_resume)
    vector = tfidf.transform([cleaned]).toarray()
    pred_cat = svc_model.predict(vector)
    return le.inverse_transform(pred_cat)[0]

def extract_keywords(text, n=10):
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:n]]

def extract_skills(text):
    SKILL_DB = ['python','java','c++','sql','excel','machine learning','deep learning',
                'tensorflow','pytorch','nlp','communication','leadership','teamwork','html','css','javascript']
    text_lower = text.lower()
    skills_found = [skill for skill in SKILL_DB if skill in text_lower]
    return skills_found

def score_resume(text):
    score = 0
    length = len(text.split())
    if length > 300:
        score += 40
    elif length > 150:
        score += 20

    keywords = extract_keywords(text)
    if len(keywords) >= 5:
        score += 30

    skills = extract_skills(text)
    if len(skills) >= 3:
        score += 30

    return min(score, 100)

def extract_entities_safe(text):
    if not SPACY_AVAILABLE:
        return {"NAME": [], "EMAIL": [], "PHONE": [], "EDUCATION": []}

    try:
        doc = nlp(text)
        entities = {"NAME": [], "EMAIL": [], "PHONE": [], "EDUCATION": []}

        # Extract names and organizations/education
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["NAME"].append(ent.text)
            if ent.label_ in ["ORG", "EDUCATION"]:
                entities["EDUCATION"].append(ent.text)

        # Extract emails
        entities["EMAIL"] = re.findall(r'\S+@\S+', text)

        # Extract phone numbers
        entities["PHONE"] = re.findall(r'\+?\d[\d -]{8,}\d', text)

        # Remove duplicates
        for k in entities:
            entities[k] = list(set(entities[k]))

        return entities
    except Exception as e:
        print("Error in NER extraction:", e)
        return {"NAME": [], "EMAIL": [], "PHONE": [], "EDUCATION": []}

# ---------------- Streamlit App ----------------
def main():
    st.set_page_config(page_title="Resume NLP Checker", page_icon="📄", layout="wide")

    # ------------- Sidebar -------------
    st.sidebar.title("🧠 Resume Checker using NLP")
    st.sidebar.info("Analyze resumes using NLP and Machine Learning to extract insights.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Developed by:** Rinkal Rahane")
    st.sidebar.markdown("**Project:** AI-based Resume Checker")
    st.sidebar.markdown("**Technology:** Streamlit, NLP, Scikit-Learn")
    st.sidebar.markdown("---")
    st.sidebar.caption("Upload a resume and optionally a job description to check category, skills, and score.")

    # ------------- Main UI -------------
    st.title("📄 Resume NLP Checker")
    st.write("Upload a resume (PDF/DOCX/TXT) to analyze category, skills, keywords, and more.")

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf","docx","txt"])
    job_desc = st.text_area("Optional: Paste Job Description for matching", height=100)

    if uploaded_file:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Resume text extracted!")

            with st.expander("📜 Show Extracted Text"):
                st.text_area("", resume_text[:3000], height=300)

            # Category prediction
            category = pred(resume_text)
            st.subheader("🔍 Predicted Category")
            st.success(category)

            # Skills and keywords
            skills = extract_skills(resume_text)
            keywords = extract_keywords(resume_text)
            st.subheader("💡 Skills Found")
            st.write(skills if skills else "No known skills found")
            st.subheader("📑 Top Keywords")
            st.write(", ".join(keywords))

            # Resume score
            score = score_resume(resume_text)
            st.subheader("📊 Resume Score")
            st.progress(score)

            # Named Entity Recognition
            entities = extract_entities_safe(resume_text)
            st.subheader("📌 Extracted Information")
            if SPACY_AVAILABLE:
                st.write("**Name:**", ", ".join(entities["NAME"]) if entities["NAME"] else "Not found")
                st.write("**Email:**", ", ".join(entities["EMAIL"]) if entities["EMAIL"] else "Not found")
                st.write("**Phone:**", ", ".join(entities["PHONE"]) if entities["PHONE"] else "Not found")
                st.write("**Education / Organizations:**", ", ".join(entities["EDUCATION"]) if entities["EDUCATION"] else "Not found")
            else:
                st.warning(".")

            # Job description match
            if job_desc:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform([resume_text, job_desc])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]*100
                st.subheader("🎯 Job Description Match")
                st.write(f"Resume matches job description by: **{similarity:.2f}%**")

        except Exception as e:
            st.error(f"Error processing resume: {str(e)}")
            st.sidebar.title("📘 About the App")
            st.sidebar.info(
                "This app uses a Machine Learning model (SVM with TF-IDF) "
                "to predict the **job category** of an uploaded resume.\n\n"
                "Upload your file → Get category → See keyword insights."
            )
            st.sidebar.markdown("----")
            st.sidebar.markdown("👨‍💻 **Created by Rinkal Rahane**")

            # Main Title
            st.title("📄 Resume Category Prediction")
            st.write("Upload your resume in PDF, DOCX, or TXT format to predict the job category.")

            uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

            if uploaded_file:
                try:
                    resume_text = handle_file_upload(uploaded_file)
                    st.success("✅ Text extracted successfully!")

                    with st.expander("📜 Show Extracted Text"):
                        st.text_area("", resume_text[:3000], height=300)

                    st.subheader("🔍 Predicted Category")
                    category = pred(resume_text)
                    st.markdown(f"<div class='result-box'><h2>{category}</h2></div>", unsafe_allow_html=True)

                    st.subheader("💡 Top Keywords from Resume")
                    keywords = extract_keywords(resume_text)
                    st.write(", ".join(keywords))

                    # Example insights
                    st.subheader("📈 Category Insights")
                    st.info(
                        f"For **{category}**, strong skills typically include programming, analysis, teamwork, and problem-solving.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
