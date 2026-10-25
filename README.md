# AI Resume Screener

An AI-powered web application that analyzes PDF resumes against a job description, ranks relevant candidates, and generates AI-assisted candidate evaluations.

## Live Demo

**Application:** https://ai-resume-screener-dhrybtqcj5wdhymitngksa.streamlit.app/

## Features

* **Resume Upload:** Upload multiple PDF resumes.
* **Text Extraction:** Extract text from uploaded resumes.
* **Vector Search:** Store resume embeddings in a FAISS index.
* **Candidate Ranking:** Retrieve and rank resumes based on a job description.
* **AI Candidate Evaluation:** Generate explanations of matching skills, missing or unverified skills, and relevant experience.
* **Results Table:** View ranked candidates and their similarity distances.
* **CSV Export:** Download candidate rankings for further review.
* **Resume Management:** View, remove, and manage stored resumes.

## Technologies Used

* Python
* Streamlit
* LangChain
* Hugging Face Sentence Transformers
* FAISS
* Groq API for cloud-based language model access
* PyPDF

## How It Works

1. Upload candidate resumes in PDF format.
2. Extract and split the resume text into smaller chunks.
3. Convert the text into vector embeddings.
4. Store the embeddings in a FAISS index.
5. Enter the job description.
6. Retrieve relevant resume sections and rank candidates.
7. Generate AI-assisted candidate evaluations.
8. Download the ranking results as a CSV file.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Pavankumar81992/AI-Resume-Screener.git
cd AI-Resume-Screener
```

### 2. Create a virtual environment

```bash
python -m venv myenv
```

Activate it on Windows:

```powershell
.\myenv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Set your Groq API key as an environment variable in PowerShell:

```powershell
$env:GROQ_API_KEY="your_actual_api_key"
```

Never commit API keys or other secrets to GitHub.

### 5. Start the application

```bash
python -m streamlit run app.py
```

## Privacy and Responsible Use

* Do not upload personal resumes or stored candidate data to a public GitHub repository.
* Configure API keys securely using environment variables or Streamlit Secrets.
* Treat similarity rankings and AI-generated explanations as decision-support information, not definitive hiring decisions.
* Review candidate evidence fairly and verify relevant qualifications before making hiring decisions.

## Author

**Mummidi Pavan Kumar**

Computer Science and Engineering Graduate | Aspiring Software Developer
