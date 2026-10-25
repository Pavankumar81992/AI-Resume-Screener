"""
Core pipeline for the AI Resume Screener.
Handles PDF loading, text chunking, and embeddings.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"


def load_pdf(pdf_file):
    """Load text from a PDF resume and preserve its original filename."""

    temp_path = Path(pdf_file.name)

    try:
        with open(temp_path, "wb") as file:
            file.write(pdf_file.getvalue())

        loader = PyPDFLoader(str(temp_path))
        documents = loader.load()

        # Preserve the original resume filename in document metadata.
        for document in documents:
            document.metadata["source"] = pdf_file.name

        return documents

    finally:
        if temp_path.exists():
            temp_path.unlink()


def chunk_documents(documents):
    """Split resume documents into smaller text chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )

    return splitter.split_documents(documents)


def create_embeddings():
    """Create the Hugging Face embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


def build_faiss(chunks, embeddings):
    """Create a FAISS vector store from document chunks."""

    if not chunks:
        raise ValueError("No document chunks were provided.")

    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )


def rank_resumes(store, job_description, k=10):
    """
    Retrieve relevant resume sections for a job description.

    Args:
        store: Existing FAISS vector store.
        job_description: Job description to search against.
        k: Maximum number of resume sections to retrieve.

    Returns:
        List of (document, distance) pairs, ordered by distance.
    """

    if not job_description or not job_description.strip():
        raise ValueError("Please provide a job description.")

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("The number of results must be a positive integer.")

    if store.index.ntotal == 0:
        return []

    # Retrieve up to k relevant resume sections.
    result_count = min(k, store.index.ntotal)

    results = store.similarity_search_with_score(
        query=job_description.strip(),
        k=result_count,
    )

    return results

def connect_llm():
    """Connect to the locally installed Ollama model."""

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )


def explain_candidate(llm, job_description, resume_text):
    """Generate a concise, evidence-based candidate evaluation."""

    prompt = ChatPromptTemplate.from_template(
        """
        You are a professional recruitment screening assistant.

        Evaluate the candidate using only the provided job description
        and resume evidence.

        JOB DESCRIPTION:
        {job_description}

        RESUME EVIDENCE:
        {resume_text}

        Follow this exact structure:

        MATCHING REQUIRED SKILLS
        - List only required skills explicitly supported by the evidence.

        MISSING OR UNVERIFIED REQUIRED SKILLS
        - List required skills not demonstrated in the evidence.
        - Do not assume the candidate lacks a skill simply because
          it is not mentioned.

        MATCHING PREFERRED SKILLS
        - List only preferred skills supported by the evidence.

        MISSING OR UNVERIFIED PREFERRED SKILLS
        - List preferred skills not demonstrated in the evidence.

        RELEVANT EXPERIENCE
        - Summarize relevant employment, internships, and projects.
        - State whether the required experience duration is demonstrated.
        - Never invent dates, responsibilities, or years of experience.

        OVERALL ASSESSMENT
        - Summarize the candidate's strongest relevant evidence.
        - Identify important gaps against the job requirements.
        - Keep the assessment balanced, concise, and factual.

        STRICT RULES:
        - Use plain text and short bullet points.
        - Do not use Markdown headings, hashtags, emojis, or decorative symbols.
        - Use only information present in the supplied evidence.
        - Never list the same skill in both matching and missing sections.
        - Keep skills, qualifications, and experience as separate criteria.
        - Treat unmentioned skills as unverified, not automatically absent.
        - Do not invent qualifications or experience.
        - Do not calculate or invent a match percentage.
        - If a section has no entries, write "None identified".
        - Avoid repeating the same information across sections.
        - Do not make a final hiring decision.

        Return only the evaluation using the specified structure.
        """
    )

    response = (prompt | llm).invoke(
        {
            "job_description": job_description,
            "resume_text": resume_text,
        }
    )

    return response.content