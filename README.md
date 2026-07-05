# Gemini-powered LangChain RAG Pipeline

This is a complete, self-contained Retrieval-Augmented Generation (RAG) pipeline built using Python, LangChain, Google Gemini API, and ChromaDB.

## File Structure

- [rag_pipeline.py](file:///home/saraskashyap/genAI_project/rag_pipeline.py): The main Python RAG pipeline script containing the ingestion, retrieval, and generation code.
- [dataset.json](file:///home/saraskashyap/genAI_project/dataset.json): A sample JSON dataset containing structured text documents.
- [requirements.txt](file:///home/saraskashyap/genAI_project/requirements.txt): Python dependency file.
- [.env.example](file:///home/saraskashyap/genAI_project/.env.example): Environment variable template for your Google API Key.


## Custom Data Ingestion

Before starting the pipeline, you can place your own custom document files (such as `.txt`, `.md`, `.pdf`, `.docx`, `.jsonld`, `.csv`, etc.) into the `data/` directory at the root of the project. The pipeline will automatically scan, parse, chunk, embed, and index these files alongside the sample `dataset.json` file.


## Setup Instructions

1. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Rename `.env.example` to `.env` and paste your Google API key:
   ```bash
   cp .env.example .env
   ```
   Add your Google API Key inside `.env`:
   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

4. **Run the RAG Pipeline CLI:**
   ```bash
   python rag_pipeline.py
   ```

## Features

- **Multi-Format Ingestion**: Automatically scans, reads, and parses JSON datasets as well as text (`.txt`), markdown (`.md`), PDF (`.pdf`), Word (`.docx`), and CSV (`.csv`) files. Documents are dynamically split into overlapping chunks (8,000 characters chunk size, 800 characters overlap) using the `RecursiveCharacterTextSplitter`.
- **Robust API Rate-Limiting**: Integrates a custom `RateLimitedEmbeddings` wrapper that batches document embeddings and sleeps between API calls (with automatic retry-backoff) to stay safely under Gemini API free-tier RPM (Requests Per Minute) and TPM (Tokens Per Minute) rate limits.
- **Vector Storage**: Indexes chunks and generates embeddings via `models/gemini-embedding-2`, storing them in a local ChromaDB instance (`chroma_db`).
- **Context-Strict Generation**: Uses `gemini-2.5-flash` with a tailored prompt specifying that the model must answer *only* from the provided context or state "I do not know" if the context is insufficient.
- **Dynamic CLI Loop**: An interactive console prompt allowing users to enter queries and view responses as well as referenced context sources.
