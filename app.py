import streamlit as st
from utils.pdf_reader import extract_text_from_pdf, chunk_text, create_embeddings, create_faiss_index
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("AI Resume Analyzer (RAG)")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_resume and "index" not in st.session_state:
    
    with st.spinner("Processing resume..."):
        text = extract_text_from_pdf(uploaded_resume)
        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)
        index = create_faiss_index(embeddings)
        
    st.session_state.index = index
    st.session_state.chunks = chunks
    st.success("✅ Resume processed and indexed successfully!")
    st.info(f"📊 Total sections indexed: {index.ntotal}")

    
if "index" in st.session_state:
    
    job_description = st.text_area("Paste Job Description", height=200)

    if job_description:
        # Retrieve relevant chunks
        query_embedding = create_embeddings([job_description])
        distances, indices = st.session_state.index.search(query_embedding, k=5)

        retrieved_text = "\n\n".join(
            [st.session_state.chunks[i] for i in indices[0]]
        )

        with st.expander("📄 Retrieved Resume Context"):
            st.write(retrieved_text)
        
        # Analyze with Groq
        prompt = f"""You are an expert ATS (Applicant Tracking System) and technical recruiter.

**RESUME CONTEXT:**
{retrieved_text}

**JOB DESCRIPTION:**
{job_description}

**TASK:**
Analyze the resume against the job description and provide a structured response:

1. **Match Score**: Rate 0-100 with brief justification
2. **Matching Skills**: List specific skills found in both resume and JD
3. **Missing Critical Skills**: Key skills in JD but not in resume
4. **Experience Alignment**: How well does experience match requirements?
5. **Recommendations**: 3-5 specific, actionable improvements to increase match score

Be specific and reference actual content from the resume. Format your response clearly with headers."""

        with st.spinner("🤖 Analyzing with AI..."):
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert ATS and technical recruiter. Provide detailed, actionable feedback."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                analysis = response.choices[0].message.content
                
                st.subheader("📊 AI Analysis")
                st.markdown(analysis)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please check your GROQ_API_KEY in .env file")