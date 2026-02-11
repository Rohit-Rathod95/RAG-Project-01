import streamlit as st
from utils.pdf_reader import extract_text_from_pdf, chunk_text, create_embeddings, create_faiss_index

st.title("AI Resume Analyzer (RAG)")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_resume:
    
    
    text = extract_text_from_pdf(uploaded_resume)
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)
    st.write("Number of Chunks:", len(chunks))
    st.write("First Chunk Preview:")
    st.write(chunks[0])
    index = create_faiss_index(embeddings)

    st.write("FAISS index created successfully!")
    st.write("Total vectors stored:", index.ntotal)

