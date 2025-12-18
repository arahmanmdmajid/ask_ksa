<p align="center">
  <img src="AskKSA_Logo.png" alt="AskKSA Logo" width="180" style="border-radius: 12px;"/>
</p>

<h1 align="center">AskKSA — Smart Bilingual Saudi Services Assistant</h1>

<p align="center">
  <strong>A modern modular RAG-powered assistant for Iqama, visa and Absher guidance.</strong>
  <br />
  Bilingual • Grounded • Fast • Modular Architecture
</p>

<p align="center">
  <a href="https://askksa.streamlit.app/">
    <img src="https://img.shields.io/badge/Live_Demo-Click_Here-brightgreen?style=for-the-badge" />
  </a>
</p>

---

🔰 **Badges**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/AI-Gemini%202.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Vector%20DB-Chroma-009688?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Embeddings-BGE--M3-6C63FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Hosted%20on-Streamlit%20Cloud-0F172A?style=for-the-badge&logo=cloudflare&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

---

## 📚 Table of Contents
- [Overview](#overview)
- [What Problem Does It Solve?](#what-problem-does-it-solve)
- [How It Solves These Problems](#how-it-solves-these-problems)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Data Ingestion & Vector DB](#data-ingestion--vector-db)
- [Modular Project Structure](#modular-project-structure)
- [Installation & Local Setup](#installation--local-setup)
- [Live Demo](#live-demo)
- [Safety & Limitations](#safety--limitations)

---

## Overview

**AskKSA** is a modern **Retrieval-Augmented Generation (RAG)** assistant providing step-by-step help for:

- Saudi Arabia **Iqama**
- Exit/Re-entry visas
- Visit visas
- Absher services
- MOI processes
- Government portal guidance

The system aims to make answers:

✔ Grounded in curated Saudi government / expat content  
✔ Bilingual (English + Urdu)  
✔ Structured, accurate, and easy to follow  
✔ Accessible through a clean Streamlit chat interface  

Under the hood, the app uses:

- **BGE-M3** sentence-transformer embeddings  
- A **Chroma** vector database for semantic retrieval  
- **Google Gemini 2.5 Flash** for answer generation  
- A modular RAG pipeline built in Python/Streamlit :contentReference[oaicite:0]{index=0}  

---

## What Problem Does It Solve?

Saudi expats and residents often struggle with:

- Conflicting information about government services  
- Portal navigation issues  
- Confusion around required documents and fees  
- Step-by-step processes scattered across multiple websites  
- Language accessibility (English/Urdu vs mostly Arabic portals)

**AskKSA**:

- Centralizes curated service information  
- Answers using *only* the curated dataset + vector DB  
- Supports English & Urdu with native RTL Urdu rendering  
- Provides fast, reliable guidance with source links so users can verify

---

## How It Solves These Problems

AskKSA uses **RAG** (Retrieval-Augmented Generation) with an **offline ingestion pipeline** and an online chat UI:

1. User asks a question (in English or Urdu).  
2. The query is embedded using **BGE-M3**.  
3. The **Chroma** collection is queried to retrieve the top-K most relevant chunks (with titles, URLs, and scores).  
4. Gemini 2.5 Flash receives a prompt that includes:
   - Language rule (English vs Urdu)  
   - System safety instructions  
   - Retrieved context (chunks + metadata)  
   - The user’s question  
5. Gemini generates a grounded answer based on the context.  
6. The UI displays:
   - The final answer  
   - The underlying sources (title, URL, similarity score)

This design keeps the model focused on verified content and makes it easy for users to inspect where an answer came from.

---

## Features

### ⭐ Core Features

- Fully bilingual (English + Urdu)  
- Auto language detection (Urdu script vs Latin)  
- Native Urdu rendering (Noto Nastaliq Urdu + RTL alignment)  
- “Helpful / Not Helpful” feedback for each answer  
- Sources sidebar with titles, links, and similarity scores  
- Persistent chat history within the session  
- Clean Streamlit chat UI with avatars  

### ⚙️ Technical Features

- Modular RAG architecture (separate ingestion, retrieval, and UI layers)  
- SentenceTransformers **BGE-M3** embeddings  
- **Chroma** vector database for semantic retrieval  
- Google Gemini 2.5 Flash model for answer generation  
- Cached model & vector DB client loading for performance  
- Centralized prompts via `prompts.py`  
- Config-driven behavior via `config.py` (model name, paths, etc.)  
- Debug panel to inspect raw retrieval results (documents, metadata, scores)  

---

## Tech Stack

### 🔹 **Frontend**
- Streamlit  
- Custom CSS (Urdu-friendly RTL rendering)  
- Google Fonts (Noto Nastaliq Urdu)

### 🔹 **Backend / AI**
- Google Gemini 2.5 Flash (via `genai` client)  
- Sentence Transformers (BAAI/bge-m3)  
- Python 3.10+

### 🔹 **RAG / Data Layer**
- **Chroma** persistent vector database (document embeddings + metadata)  
- Offline ingestion scripts:
  - Scraping (Playwright + BeautifulSoup)  
  - Markdown frontmatter parsing (`title`, `source_url`, `scraped_at`)  
  - Chunking & metadata enrichment  
  - Embedding with BGE-M3 and writing to Chroma  

(Older FAISS + `chunks.json` / `chunks_metadata.json` artifacts have been replaced by this Chroma-based workflow.)

---

## Architecture

```text
           ┌──────────────────────────────┐
           │     User Question (Eng/Urdu) │
           └──────────────────────────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │  Embed (BGE-M3 Model)  │
             └────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │  Chroma Vector DB Retrieval    │
         │ (Top-K context + metadata)     │
         └────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────┐
       │ Prompt: Lang rule + safety + context │
       └──────────────────────────────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Gemini 2.5 Flash Model │
              └────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │ Answer + Sources (UI layer)  │
           └──────────────────────────────┘
```

---

## Modular Project Structure

To ensure scalability and maintainability, AskKSA now uses a **clean modular architecture**:

```text
ASK_KSA/
│
├── .streamlit/           # Streamlit config + secrets.toml
├── data/                 # Markdown publications and source data
├── vector_db/            # Persistent Chroma vector database (generated)
│
├── app.py                # Streamlit UI (chat, sidebar, debug tools)
├── config.py             # Central configuration (paths, model names, constants)
├── data_loader.py        # Loads embedding model + Chroma collection
├── llm_client.py         # Gemini client + unified LLM interface
├── prompts.py            # Prompt templates and language rules
├── rag_core.py           # Retrieval, context building, answer generation
├── scrapping.py          # Scraping / Playwright scripts (offline data collection)
├── utils.py              # Helpers (markdown loading, cleaning, slugify, seeding)
├── vector_db_ingest.py   # Offline ingestion: build/update Chroma vector DB
│
├── requirements.txt      # Python dependencies
├── LICENSE.txt           # License
└── README.md             # Documentation
```

This structure allows:

* Swapping datasets by re-running the ingestion script
* Changing LLM models from a single config location
* Editing prompts without touching retrieval logic
* Adding languages or new domains without breaking the core pipeline

---

## Installation & Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/arahmanmdmajid/DS_AI_11
cd DS_AI_11/askksa
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### 4. Add your Gemini API key

* Rename the **`sample_secrets.toml`** file to **`secrets.toml`** and add your API key:

  * **Path**: `.streamlit/secrets.toml`
  * Format:

    ```toml
    [general]
    GOOGLE_API_KEY = "your_key_here"
    ```

* Or you can also add your Gemini API key as an environment variable

```bash
set GOOGLE_API_KEY="your_key_here"
```

### 5. (Optional) Build / refresh the vector DB

If you want to regenerate the vector database locally:

```bash
python vector_db_ingest.py
```

This will read the markdown publications, chunk and embed them, and populate the `vector_db/` directory.


### 5. Run locally

```bash
streamlit run app.py
```

---

## Live Demo

🚀 **Try AskKSA here:**
👉 [https://askksa.streamlit.app/](https://askksa.streamlit.app/)

---

## Safety & Limitations

AskKSA is an informational assistant, not an official Saudi government service and not a legal advisor. It focuses on common Iqama/visa/Absher tasks and uses a curated, periodically updated dataset, but:

* Some answers may become outdated if policies change.
* Urdu wording and nuance may occasionally be less polished than English.
* The app depends on an external LLM API (Gemini), which can sometimes be slow or temporarily unavailable.

For high-impact or time-sensitive decisions (fines, residency violations, disputes, etc.), always double-check with official government portals or a qualified professional before acting on any answer.
