"""
AI Resume Screener
Home: Upload and store PDF resumes.
Analyze: Search resumes against a job description.
About: Project information.
"""

import json
import shutil
import pandas as pd
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS

from pipeline import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    load_pdf,
    chunk_documents,
    create_embeddings,
    build_faiss,
    rank_resumes,
    connect_llm,
    explain_candidate,
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
)


# --------------------------------------------------
# File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "stored_faiss"
MANIFEST = INDEX_DIR / "manifest.json"


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading the embedding model...")
def embeddings_model():
    return create_embeddings()


# --------------------------------------------------
# Save FAISS index and resume information
# --------------------------------------------------

def save_index(store, names):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    store.save_local(str(INDEX_DIR))

    manifest_data = {
        "resumes": names,
        "embedding_model": EMBEDDING_MODEL,
    }

    MANIFEST.write_text(
        json.dumps(manifest_data, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------
# Load previously saved FAISS index
# --------------------------------------------------

def load_saved_index():
    if not MANIFEST.exists():
        return None, []

    try:
        manifest_data = json.loads(
            MANIFEST.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        st.error("The resume manifest could not be read.")
        return None, []

    if manifest_data.get("embedding_model") != EMBEDDING_MODEL:
        st.error(
            "The saved index uses a different embedding model. "
            "Please rebuild the index."
        )
        return None, []

    if not (
        (INDEX_DIR / "index.faiss").exists()
        and (INDEX_DIR / "index.pkl").exists()
    ):
        st.warning("The saved FAISS index is incomplete.")
        return None, []

    try:
        store = FAISS.load_local(
            str(INDEX_DIR),
            embeddings_model(),
            allow_dangerous_deserialization=True,
        )
    except Exception as error:
        st.error(f"Could not load the FAISS index: {error}")
        return None, []

    return store, manifest_data.get("resumes", [])


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

st.sidebar.title("📄 AI Resume Screener")

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

if st.session_state.pop("navigate_to_analysis", False):
    st.session_state["page"] = "🔎 Analyze Resumes"

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔎 Analyze Resumes",
        "ℹ️ About",
    ],
    key="page",
)

st.sidebar.caption(
    "LangChain · HuggingFace · FAISS · RAG · Ollama"
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "🏠 Home":

    st.title("🏠 Home — Upload and Store Resumes")

    st.write(
        "Upload candidate resumes in PDF format. "
        "The application extracts their text, creates embeddings, "
        "and stores them in a searchable FAISS index."
    )

    # --------------------------------------------------
    # UPLOAD RESUMES
    # --------------------------------------------------

    if "upload_widget_key" not in st.session_state:
        st.session_state.upload_widget_key = 0

    uploaded_files = st.file_uploader(
        "Choose PDF resumes",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"resume_uploader_{st.session_state.upload_widget_key}",
    )

    col1, col2 = st.columns(2)

    with col1:
        upload_clicked = st.button(
            "📥 Upload & Store in FAISS",
            type="primary",
        )

    with col2:
        if st.button("🧹 Clear Selected Files"):
            st.session_state.upload_widget_key += 1
            st.rerun()

    if upload_clicked:

        if not uploaded_files:
            st.warning("Please upload at least one PDF resume.")

        else:
            existing_store, existing_names = load_saved_index()
            existing_names = list(existing_names)

            new_chunks = []
            new_names = []

            progress = st.progress(0)

            for number, uploaded_file in enumerate(uploaded_files):

                filename = Path(uploaded_file.name).name

                if filename in existing_names or filename in new_names:
                    st.info(f"Skipped duplicate filename: {filename}")

                else:
                    try:
                        documents = load_pdf(uploaded_file)

                        if not documents or not any(
                            document.page_content.strip()
                            for document in documents
                        ):
                            st.warning(
                                f"No extractable text found in {filename}."
                            )

                        else:
                            for document in documents:
                                document.metadata["source"] = filename

                            chunks = chunk_documents(documents)

                            if chunks:
                                new_chunks.extend(chunks)
                                new_names.append(filename)
                            else:
                                st.warning(
                                    f"No text chunks created for {filename}."
                                )

                    except Exception as error:
                        st.error(
                            f"Could not process {filename}: {error}"
                        )

                progress.progress((number + 1) / len(uploaded_files))

            if new_chunks:
                try:
                    with st.spinner(
                        "Creating embeddings and updating FAISS..."
                    ):
                        embeddings = embeddings_model()

                        if existing_store is not None:
                            new_store = build_faiss(
                                new_chunks,
                                embeddings,
                            )
                            existing_store.merge_from(new_store)
                            final_store = existing_store
                        else:
                            final_store = build_faiss(
                                new_chunks,
                                embeddings,
                            )

                        combined_names = existing_names + new_names
                        save_index(final_store, combined_names)

                    st.success(
                        f"Successfully stored {len(new_names)} resume(s) "
                        f"and indexed {len(new_chunks)} text chunk(s)."
                    )

                    st.session_state.upload_widget_key += 1
                    st.rerun()

                except Exception as error:
                    st.error(
                        f"Could not build the resume index: {error}"
                    )

            else:
                st.info("No new resumes were added.")

    st.divider()

    # --------------------------------------------------
    # STORED RESUME MANAGEMENT
    # --------------------------------------------------

    st.subheader("📚 Stored Resumes")

    current_store, stored_names = load_saved_index()

    if stored_names:

        st.write(f"**Total stored resumes: {len(stored_names)}**")

        selected_resume = st.selectbox(
            "Choose a resume to remove",
            options=stored_names,
            key="resume_to_remove",
        )

        if st.button("🗑️ Remove Selected Resume"):

            try:
                if current_store is None:
                    st.error(
                        "The saved FAISS index could not be loaded."
                    )

                else:
                    remaining_documents = [
                        document
                        for document in current_store.docstore._dict.values()
                        if document.metadata.get("source") != selected_resume
                    ]

                    remaining_names = [
                        name
                        for name in stored_names
                        if name != selected_resume
                    ]

                    if remaining_documents:
                        with st.spinner("Rebuilding the FAISS index..."):
                            rebuilt_store = build_faiss(
                                remaining_documents,
                                embeddings_model(),
                            )

                            save_index(
                                rebuilt_store,
                                remaining_names,
                            )

                    else:
                        if INDEX_DIR.exists():
                            shutil.rmtree(INDEX_DIR)

                    st.session_state.pop("resume_to_remove", None)

                    st.success(
                        f"Removed {selected_resume} successfully."
                    )
                    st.rerun()

            except Exception as error:
                st.error(f"Could not remove the resume: {error}")

        st.divider()

        if st.button("⚠️ Clear All Stored Resumes"):
            st.session_state.confirm_clear_all = True

        if st.session_state.get("confirm_clear_all", False):

            st.warning(
                "This will permanently remove all stored resumes "
                "and the FAISS index from this application."
            )

            confirm_col, cancel_col = st.columns(2)

            with confirm_col:
                if st.button("Yes, Delete Everything"):

                    try:
                        if INDEX_DIR.exists():
                            for filename in (
                                "index.faiss",
                                "index.pkl",
                                "manifest.json",
                            ):
                                file_path = INDEX_DIR / filename

                                if file_path.exists():
                                    file_path.unlink()

                        st.session_state.confirm_clear_all = False
                        st.session_state.pop("resume_to_remove", None)
                        st.session_state.upload_widget_key += 1

                        st.success("All stored resumes have been cleared.")
                        st.rerun()

                    except OSError as error:
                        st.error(
                            f"Could not clear the stored resumes: {error}"
                        )

            with cancel_col:
                if st.button("Cancel Deletion"):
                    st.session_state.confirm_clear_all = False
                    st.rerun()

        st.divider()

        for number, filename in enumerate(stored_names, start=1):
            st.write(f"{number}. {filename}")

        st.divider()

        if st.button(
            "🔎 Proceed to Analyze Resumes",
            type="primary",
            key="proceed_to_analysis",
        ):
            st.session_state["upload_complete"] = False
            st.session_state["navigate_to_analysis"] = True
            st.rerun()

    else:
        st.info("No resumes have been stored yet.")

        if st.session_state.get("confirm_clear_all", False):
            if st.button("Cancel Deletion"):
                st.session_state.confirm_clear_all = False
                st.rerun()

## --------------------------------------------------
# ANALYZE RESUMES PAGE
# --------------------------------------------------

elif page == "🔎 Analyze Resumes":

    st.title("🔎 Analyze Resumes")

    st.write(
        "Compare stored resumes with a job description "
        "and review the most relevant candidates."
    )

    # Initialize session state to preserve results across reruns.
    if "ranked_candidates" not in st.session_state:
        st.session_state.ranked_candidates = []

    if "analysis_job_description" not in st.session_state:
        st.session_state.analysis_job_description = ""

    if "candidate_explanations" not in st.session_state:
        st.session_state.candidate_explanations = {}

    job_description = st.text_area(
        "Enter the job description",
        height=200,
        placeholder=(
            "Example: Python developer with SQL, FastAPI, "
            "REST API development, and two years of experience."
        ),
        key="job_description_input",
    )

    top_k = st.number_input(
        "Number of candidates to shortlist",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        key="top_k_input",
    )

    # Search candidates.
    if st.button("🔍 Analyze Candidates", type="primary"):

        if not job_description.strip():
            st.warning("Please enter a job description.")

        else:
            store, stored_names = load_saved_index()

            if store is None:
                st.warning(
                    "No valid saved index was found. "
                    "Go to Home and upload resumes first."
                )

            else:
                try:
                    with st.spinner("Searching relevant resumes..."):

                        results = rank_resumes(
                            store,
                            job_description,
                            k=min(
                                store.index.ntotal,
                                max(top_k * 10, top_k),
                            ),
                        )

                    candidates = {}

                    for document, distance in results:

                        filename = document.metadata.get(
                            "source",
                            "Unknown resume",
                        )

                        if filename not in candidates:
                            candidates[filename] = {
                                "distance": float(distance),
                                "texts": [],
                            }

                        candidates[filename]["distance"] = min(
                            candidates[filename]["distance"],
                            float(distance),
                        )

                        candidates[filename]["texts"].append(
                            document.page_content
                        )

                    ranked_candidates = sorted(
                        candidates.items(),
                        key=lambda item: item[1]["distance"],
                    )[:top_k]

                    # Save results so they survive Streamlit reruns.
                    st.session_state.ranked_candidates = ranked_candidates
                    st.session_state.analysis_job_description = (
                        job_description
                    )
                    st.session_state.candidate_explanations = {}

                except Exception as error:
                    st.error(f"Analysis failed: {error}")

    # Display saved search results after every rerun.
    ranked_candidates = st.session_state.ranked_candidates

    if ranked_candidates:

        st.subheader("📊 Ranked Candidates")

        ranking_data = [
            {
                "Rank": rank,
                "Resume": filename,
                "FAISS Distance": round(details["distance"], 4),
            }
            for rank, (filename, details) in enumerate(
                ranked_candidates,
                start=1,
            )
        ]

        st.dataframe(
            ranking_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(
                    "Rank",
                    width="small",
                    format="%d",
                ),
                "Resume": st.column_config.TextColumn(
                    "Resume",
                    width="large",
                ),
                "FAISS Distance": st.column_config.NumberColumn(
                    "FAISS Distance",
                    format="%.4f",
                ),
            },
        )

        st.caption(
            "Lower FAISS distance generally indicates greater similarity. "
            "Distance is not a match percentage or a hiring decision."
        )

        ranking_df = pd.DataFrame(ranking_data)

        st.download_button(
            label="📥 Download Ranked Candidates (CSV)",
            data=ranking_df.to_csv(index=False).encode("utf-8"),
            file_name="ranked_candidates.csv",
            mime="text/csv",
        )

        st.divider()

        for rank, (filename, details) in enumerate(
            ranked_candidates,
            start=1,
        ):

            with st.expander(
                f"#{rank} — {filename}",
                expanded=(rank == 1),
            ):

                st.write(
                    "**Retrieved distance:** "
                    f"{details['distance']:.4f}"
                )

                st.caption(
                    "Lower distance generally indicates greater "
                    "similarity. This is not a match percentage."
                )

                st.write("**Relevant resume sections**")

                resume_text = "\n\n".join(details["texts"])

                st.text(resume_text)

                               # Store the explanation under a unique candidate key.
                explanation_key = f"{rank}_{filename}"

                if st.button(
                    "🤖 Explain Candidate",
                    key=f"explain_{explanation_key}",
                ):
                    try:
                        llm = connect_llm()

                        with st.spinner(
                            "Generating candidate explanation..."
                        ):
                            explanation = explain_candidate(
                                llm,
                                st.session_state.analysis_job_description,
                                resume_text,
                            )

                        st.session_state.candidate_explanations[
                            explanation_key
                        ] = explanation

                    except Exception as error:
                        st.error(
                            f"Could not generate explanation: {error}"
                        )

                                # Display the explanation only when it exists.
                if explanation_key in st.session_state.candidate_explanations:

                    st.markdown("#### AI Candidate Evaluation")

                    st.markdown(
                        """
                        <style>
                        .candidate-evaluation {
                            padding: 18px;
                            border: 1px solid #444;
                            border-radius: 10px;
                            line-height: 1.7;
                            font-size: 15px;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.container(border=True):
                        st.markdown(
                            st.session_state.candidate_explanations[
                                explanation_key
                            ]
                        )

    elif st.session_state.analysis_job_description:
        st.info("No matching resume sections were found.")
# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "ℹ️ About":

    st.title("ℹ️ About AI Resume Screener")

    st.write(
        """
        AI Resume Screener is an intelligent resume-search and analysis
        application designed to help recruiters identify candidates
        relevant to a specific job description.

        The application uses text embeddings, FAISS similarity search,
        and a local language model to retrieve relevant resume content,
        rank candidates, and generate AI-powered candidate evaluations.
        """
    )

    st.divider()

    # --------------------------------------------------
    # KEY FEATURES
    # --------------------------------------------------

    st.subheader("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **📄 Resume Management**

            - Upload multiple PDF resumes.
            - Store resume content in a FAISS index.
            - View and remove stored resumes.

            **🔎 Resume Search**

            - Enter a job description.
            - Retrieve relevant resume sections.
            - Rank candidates using similarity search.
            """
        )

    with col2:
        st.markdown(
            """
            **🤖 AI Candidate Evaluation**

            - Generate candidate explanations.
            - Review matching and unverified skills.
            - Examine relevant experience.

            **📊 Candidate Review**

            - Compare retrieved candidates.
            - Review rankings and supporting resume content.
            - Use AI evaluations to support recruiter decisions.
            """
        )

    st.divider()

    # --------------------------------------------------
    # TECHNOLOGIES USED
    # --------------------------------------------------

    st.subheader("🛠️ Technologies Used")

    technologies = [
        (
            "Python",
            "Core programming language for the application."
        ),
        (
            "Streamlit",
            "Builds the interactive web interface."
        ),
        (
            "LangChain",
            "Connects document processing, vector search, and language models."
        ),
        (
            "Sentence Transformers",
            "Converts text into numerical embeddings."
        ),
        (
            "FAISS",
            "Stores and searches vectors using similarity search."
        ),
        (
            "Ollama",
            "Runs the language model locally."
        ),
        (
            "Llama 3.2",
            "Generates AI-powered candidate explanations."
        ),
        (
            "PyPDF",
            "Extracts text from PDF resumes."
        ),
    ]

    for technology, description in technologies:
        st.markdown(f"**{technology}** — {description}")

    st.divider()

    # --------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------

    st.subheader("⚙️ How It Works")

    workflow_steps = [
        ("01", "Upload Resumes", "Upload PDF resumes through the Home page."),
        ("02", "Extract Text", "Extract text from the uploaded documents."),
        ("03", "Split Documents", "Divide resume text into smaller chunks."),
        ("04", "Generate Embeddings", "Convert text chunks into vector representations."),
        ("05", "Store in FAISS", "Save the vectors for similarity search."),
        ("06", "Enter Job Description", "Provide the job requirements for screening."),
        ("07", "Retrieve Candidates", "Find and rank candidates using resume similarity."),
        ("08", "Evaluate Candidates", "Generate AI explanations for selected candidates."),
    ]

    for number, title, description in workflow_steps:
        with st.container(border=True):
            st.markdown(f"**{number} | {title}**")
            st.caption(description)

    st.divider()

    # --------------------------------------------------
    # RESPONSIBLE USE
    # --------------------------------------------------

    st.subheader("⚖️ Responsible Use")

    st.info(
        """
        AI Resume Screener is designed to support the initial resume
        screening process. Similarity rankings and AI-generated
        explanations are not definitive judgments of candidate suitability.

        Recruiters should verify the available resume evidence, evaluate
        candidates fairly, and make final hiring decisions themselves.
        """
    )

    st.caption(
        "AI Resume Screener | Resume Search, Ranking & AI-Assisted Evaluation"
    )