import streamlit as st
from utils.pdf_reader import extract_text_from_pdf, chunk_text,create_embeddings, create_faiss_index
from dotenv import load_dotenv
import os
load_dotenv()
import ollama

st.title("AI Resume Analyzer (RAG)")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_resume and "index" not in st.session_state:
    
    text = extract_text_from_pdf(uploaded_resume)
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)
    index = create_faiss_index(embeddings)
    st.session_state.index = index
    st.session_state.chunks = chunks
    st.success("Resume processed and indexed successfully!")


    st.write("FAISS index created successfully!")
    st.write("Total vectors stored:", index.ntotal)
    
    
if "index" in st.session_state:
    
    job_description = st.text_area("Paste Job Description")

    if job_description:
        query_embedding = create_embeddings([job_description])
        distances, indices = st.session_state.index.search(query_embedding, k=3)

        retrieved_text = "\n\n".join(
            [st.session_state.chunks[i] for i in indices[0]]
        )

        st.subheader("Retrieved Resume Context")
        st.write(retrieved_text)
        
        
        prompt = f"""
        You are an expert technical recruiter.
        Resume Context:
        {retrieved_text}    
        Job Description:
        {job_description}

        Provide:
        1. Match Score (0-100)
        2. Matching Skills
        3. Missing Skills
        4. Suggestions to improve resume
        """

        response = ollama.chat(
        model='llama3',
        messages=[
                    {'role': 'user', 'content': prompt}
                ]
            )
        analysis = response['message']['content']
        st.subheader("AI Analysis")
        st.write(analysis)
        st.subheader("AI Analysis")

