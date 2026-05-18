# AI Resume Analyzer (RAG)

This project is a Streamlit application that analyzes a candidate resume against a job description using a retrieval-augmented generation (RAG) workflow. It extracts text from an uploaded PDF resume, splits the content into chunks, creates embeddings, indexes them with FAISS, retrieves the most relevant resume sections for a job description, and asks a Groq-hosted LLM to produce an ATS-style analysis.

## Features

- Upload a resume in PDF format
- Extract and chunk resume text
- Generate embeddings with `sentence-transformers`
- Build a FAISS vector index for similarity search
- Retrieve the most relevant resume sections for a job description
- Generate a structured match analysis with Groq

## Tech Stack

- **Frontend/UI:** Streamlit
- **PDF parsing:** pypdf
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector search:** FAISS
- **LLM inference:** Groq API (`llama-3.3-70b-versatile`)
- **Environment variables:** python-dotenv

## Project Structure

```text
RAG-Project-01/
├── app.py
└── utils/
    └── pdf_reader.py
```

- `app.py` contains the Streamlit interface and Groq-powered analysis flow.
- `utils/pdf_reader.py` contains helper functions for PDF text extraction, chunking, embedding creation, and FAISS indexing.

## How It Works

1. A user uploads a PDF resume in the Streamlit app.
2. The app extracts the resume text with `pypdf`.
3. The text is split into overlapping chunks.
4. Each chunk is converted into embeddings with a sentence-transformer model.
5. The embeddings are stored in a FAISS index.
6. When a job description is entered, the app embeds the query and retrieves the top matching resume chunks.
7. The retrieved context and job description are sent to the Groq model for a structured resume-vs-job analysis.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rohit-Rathod95/RAG-Project-01.git
cd RAG-Project-01
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

This repository does not currently include a dependency manifest, so install the packages used by the code manually:

```bash
pip install streamlit python-dotenv groq pypdf sentence-transformers faiss-cpu numpy
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

## Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Usage

1. Start the Streamlit app.
2. Upload a PDF resume.
3. Wait for the app to process and index the resume.
4. Paste a job description into the text area.
5. Review the retrieved resume context and AI-generated analysis.

## Output

The application generates:

- A match score
- Matching skills
- Missing critical skills
- Experience alignment feedback
- Recommendations to improve the resume for the target role

## Current Limitations

- Only PDF resumes are supported.
- The repository does not yet include a `requirements.txt` or automated tests.
- Resume chunking is character-based and may split sections mid-sentence.
- The app keeps the FAISS index in Streamlit session state for the active session only.

## Future Improvements

- Add a dependency file such as `requirements.txt`
- Add automated tests and validation checks
- Support additional file formats
- Improve chunking and retrieval quality
- Persist vector indexes across sessions
