import os
import faiss
import pickle
import numpy as np
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import re
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# ✅ Configure Google Gemini API
genai.configure(api_key=api_key)

# ✅ Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Define Paths for Each Grade Separately (Updated for GitHub)
FAISS_INDEX_PATHS = {
    "9th": "biology_9th.faiss",
    "10th": "biology_10th.faiss",
    "11th": "biology_11th.faiss",
    "12th": "biology_12th.faiss",
}

CHUNKS_PATHS = {
    "9th": "biology_9th_chunks.pkl",
    "10th": "biology_10th_chunks.pkl",
    "11th": "biology_11th_chunks.pkl",
    "12th": "biology_12th_chunks.pkl",
}

METADATA_PATHS = {
    "9th": "biology_9th_metadata.pkl",
    "10th": "biology_10th_metadata.pkl",
    "11th": "biology_11th_metadata.pkl",
    "12th": "biology_12th_metadata.pkl",
}


# ✅ Retrieve Relevant Text Using FAISS
def retrieve_relevant_text(query, grade, top_k=5):
    retrieved_text, retrieved_metadata = [], []

    try:
        index = faiss.read_index(FAISS_INDEX_PATHS[grade])
        with open(CHUNKS_PATHS[grade], "rb") as f:
            chunks = pickle.load(f)
        with open(METADATA_PATHS[grade], "rb") as f:
            metadata = pickle.load(f)

        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(query_embedding, top_k)

        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                retrieved_text.append(chunks[idx])
                retrieved_metadata.append(metadata[idx])

    except Exception:
        pass  # Skip errors in FAISS loading

    return retrieved_text, retrieved_metadata


# ✅ Function to Refine Answer Using Gemini AI
def refine_answer_with_gemini(query, retrieved_text, grade):
    if not retrieved_text:
        input_text = f"""
        You are a biology teacher for grade {grade}. The student asked:

        "{query}"

        **Instructions:**
        - Explain in simple and clear terms.
        - If it's a conceptual question, provide a detailed yet easy-to-understand response.
        - If possible, give a real-world example.

        **Final Answer:**
        """
    else:
      input_text = f"""
You are a highly skilled biology teacher with expertise in academic textbooks. Below is an excerpt from the official biology textbook that may contain the answer:

📖 **Textbook Content:**  
{retrieved_text}

📌 **Student's Question:** "{query}"

🔍 **Instructions:**  
- Prioritize answering **only** from the textbook content above.  
- Summarize and explain the textbook information **without copying verbatim**.  
- Ensure the response is **clear, well-structured, and student-friendly**.  
- If the textbook does **not** provide a full answer, expand logically based on **scientific principles** (but keep it relevant to the textbook).  
- If applicable, **add a real-world example** to make it relatable.  

🎯 **Final Answer:**
"""


    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = model.generate_content(input_text)
        return response.text.strip() if response else "I couldn't generate an answer."
    except ValueError:
        return "I couldn't generate an answer."


# ✅ Function to Get Answer
def get_biology_answer(query, grade):
    retrieved_text, retrieved_metadata = retrieve_relevant_text(query, grade)

    if retrieved_text:
        best_answer = refine_answer_with_gemini(query, retrieved_text[0], grade)
        sources = [
            f"📚 *Biology {grade}* - 📄 Page {meta.get('page', 'Unknown')} - 🏷 Chapter {meta.get('chapter_number', 'Unknown')}: {meta.get('chapter_name', 'Unknown')}"
            for meta in retrieved_metadata
        ]
    else:
        best_answer = refine_answer_with_gemini(query, "", grade)
        sources = ["🌐 AI-generated response (Conceptual Question)"]

    return best_answer, sources


# ✅ Function to Generate Question Paper
def generate_question_paper(grade, chapter, topic, mcq_count, short_q_count, long_q_count, total_marks):
    chapter_query = f"{chapter} {topic}" if topic else chapter
    retrieved_text, _ = retrieve_relevant_text(chapter_query, grade, top_k=10)

    if not retrieved_text:
        return f"AI-generated questions for grade {grade}, as no textbook content was found."

    input_prompt = f"""
    You are a biology teacher. Below is a summary of textbook content:

    {retrieved_text}

    Generate a structured question paper.

    *MCQs:* {mcq_count}  
    *Short Questions:* {short_q_count}  
    *Long Questions:* {long_q_count}  

    Provide solutions at the end.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = model.generate_content(input_prompt)
        return response.text.strip() if response else "Error generating question paper."
    except ValueError:
        return "❌ Question paper could not be generated due to content restrictions."


# ✅ Streamlit UI with Page Navigation
st.set_page_config(page_title="Biology Assistant", page_icon="📘", layout="wide")

# ✅ Sidebar Navigation
st.sidebar.title("🔀 Navigation")
page = st.sidebar.radio("Go to:", ["🔍 Question Answering", "📝 Question Paper Generator"])

if page == "🔍 Question Answering":
    st.title("📘 Biology Chatbot - Question Answering")

    grade_input = st.selectbox("Select Your Grade", ["9th", "10th", "11th", "12th"], key="qa_grade")
    user_query = st.text_input("Enter your biology question:", placeholder="e.g. What is photosynthesis?")

    if user_query:
        answer, sources = get_biology_answer(user_query, grade_input)
        st.markdown(f"### 📕 Answer:\n{answer}")
        st.markdown("### 📖 Sources:")
        for source in sources:
            st.markdown(f"- {source}")

elif page == "📝 Question Paper Generator":
    st.title("📝 Generate a Custom Question Paper")

    grade_input_qp = st.selectbox("Select Grade", ["9th", "10th", "11th", "12th"], key="qp_grade")
    chapter_input = st.text_input("Enter Chapter Name:")
    topic_input = st.text_input("Enter Specific Topic (Optional):")
    mcq_count = st.number_input("Number of MCQs:", min_value=0, max_value=50, value=5)
    short_q_count = st.number_input("Number of Short Questions:", min_value=0, max_value=20, value=5)
    long_q_count = st.number_input("Number of Long Questions:", min_value=0, max_value=10, value=3)

    if st.button("Generate Question Paper"):
        question_paper = generate_question_paper(grade_input_qp, chapter_input, topic_input, mcq_count, short_q_count,
                                                 long_q_count, 100)
        st.markdown("### 📄 Generated Question Paper")
        st.write(question_paper)
