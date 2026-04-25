# 🏥 Post-Discharge Medical AI Assistant (Multi-Agent RAG)

> ⚠️ **Disclaimer:** This is an AI assistant for **educational purposes only**. Always consult qualified healthcare professionals for medical advice. This system uses dummy patient data and is not intended for clinical use.

---

## 📌 Overview

A **multi-agent AI system** built as a proof of concept (POC) for post-discharge patient care. The system uses **Retrieval-Augmented Generation (RAG)** over nephrology reference materials, a **two-agent architecture** (Receptionist + Clinical), and a **web search fallback** to answer patient queries intelligently — all through a simple Streamlit interface backed by a FastAPI server.

Built as part of the **DataSmith AI – GenAI Intern Assignment**.

---

## 🎯 Features

- 🤖 **Multi-Agent Architecture** — Receptionist Agent handles intake & routing; Clinical Agent handles medical queries
- 📄 **RAG over Nephrology PDFs** — Semantic search with FAISS + sentence-transformer embeddings
- 🔍 **Web Search Fallback** — Queries outside reference materials are answered via live web search
- 🗂️ **25+ Dummy Patient Reports** — Structured JSON discharge records with diagnosis, medications, dietary restrictions, and follow-up info
- 🔗 **Tool-Based Patient Retrieval** — Dedicated tool for patient lookup by name with error handling
- 📝 **Comprehensive Logging** — Full interaction logs with timestamps, agent handoffs, and retrieval results
- 🌐 **Streamlit UI + FastAPI Backend** — Clean interface with a robust API layer

---

## 🏗️ Architecture

```
User (Streamlit UI)
        │
        ▼
  FastAPI Backend
        │
        ▼
 Receptionist Agent  ──── Patient Data Retrieval Tool ──── JSON/SQLite DB
        │
        │  (medical query detected)
        ▼
  Clinical AI Agent
     ├── RAG Tool ──────── FAISS Vector Store ──── Nephrology PDFs
     ├── Web Search Tool ─ Live Web Search
     └── Logging System ── Log File (timestamped)
```

### Agent Roles

| Agent | Responsibility |
|---|---|
| **Receptionist Agent** | Greets patient, fetches discharge report by name, asks follow-up questions, routes medical queries |
| **Clinical AI Agent** | Answers medical questions using RAG, falls back to web search, provides citations, logs interactions |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Multi-Agent Framework** | LangGraph |
| **LLM** | HuggingFace (sentence-transformers for embeddings) |
| **Vector Database** | FAISS |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Patient Data Storage** | JSON / SQLite |
| **Web Search** | Integrated search tool in Clinical Agent |
| **Language** | Python |

---

## 📁 Project Structure

```
multi-agent/
├── agents/
│   ├── receptionist_agent.py     # Receptionist Agent logic
│   └── clinical_agent.py         # Clinical Agent with RAG + web search
├── tools/
│   ├── patient_retrieval.py      # Patient lookup tool
│   ├── rag_tool.py               # FAISS-based RAG tool
│   └── web_search_tool.py        # Web search fallback tool
├── data/
│   ├── patient_reports/          # 25+ dummy discharge reports (JSON)
│   └── nephrology_reference/     # Reference PDFs for RAG
├── vector_store/
│   └── faiss_index/              # Pre-built FAISS embeddings
├── logs/
│   └── interactions.log          # Timestamped interaction logs
├── backend/
│   └── main.py                   # FastAPI server
├── frontend/
│   └── app.py                    # Streamlit UI
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/jaidh01/multi-agent.git
cd multi-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
HUGGINGFACE_API_KEY=your_hf_key_here
WEB_SEARCH_API_KEY=your_search_api_key_here   # e.g. Tavily or SerpAPI
```

### 5. Build the vector store

```bash
python tools/rag_tool.py --build
```

### 6. Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 7. Launch the Streamlit UI

```bash
streamlit run frontend/app.py
```

---

## 💬 Example Interaction

```
System:   "Hello! I'm your post-discharge care assistant. What's your name?"

Patient:  "John Smith"

Receptionist Agent: [Fetches discharge report for John Smith]
          "Hi John! I found your discharge report from January 15th for
           Chronic Kidney Disease Stage 3. How are you feeling today?
           Are you following your medication schedule?"

Patient:  "I'm having swelling in my legs. Should I be worried?"

Receptionist Agent: "This sounds like a medical concern. Let me connect
                     you with our Clinical AI Agent."

Clinical Agent: "Based on your CKD diagnosis and nephrology guidelines,
                 leg swelling can indicate fluid retention... [RAG response
                 with citations from nephrology reference]"

Patient:  "What's the latest research on SGLT2 inhibitors for kidney disease?"

Clinical Agent: "This requires recent information. Let me search for you...
                 According to recent medical literature: [Web search results
                 with source cited]"
```

---

## 📋 Patient Report Schema

```json
{
  "patient_name": "John Smith",
  "discharge_date": "2024-01-15",
  "primary_diagnosis": "Chronic Kidney Disease Stage 3",
  "medications": ["Lisinopril 10mg daily", "Furosemide 20mg twice daily"],
  "dietary_restrictions": "Low sodium (2g/day), fluid restriction (1.5L/day)",
  "follow_up": "Nephrology clinic in 2 weeks",
  "warning_signs": "Swelling, shortness of breath, decreased urine output",
  "discharge_instructions": "Monitor blood pressure daily, weigh yourself daily"
}
```

---

## 📊 Logging

All interactions are logged to `logs/interactions.log` with the following information:

- Timestamps for every agent action
- Patient data retrieval attempts and results
- Agent handoff events (Receptionist → Clinical)
- RAG retrieval results and source citations
- Web search fallback triggers and results
- Error cases (patient not found, ambiguous names, etc.)

---

## ✅ Assignment Checklist

- [x] 25+ dummy patient reports created
- [x] Nephrology reference materials processed and embedded
- [x] Receptionist Agent implemented
- [x] Clinical AI Agent with RAG implemented
- [x] Patient data retrieval tool with error handling
- [x] Web search tool integration
- [x] Comprehensive logging system with timestamps
- [x] Streamlit web interface
- [x] Agent handoff mechanism
- [x] Source citations in RAG responses
- [x] Medical disclaimer added

---

## 🚧 Known Limitations

- Uses dummy patient data only — not suitable for real clinical use
- Web search results are not medically verified
- LLM responses may occasionally hallucinate — always cross-check with a healthcare professional

---

## 👤 Author

**Jai Dhingra**
- GitHub: [@jaidh01](https://github.com/jaidh01)
- LinkedIn: [linkedin.com/in/jai-dhingra](https://linkedin.com/in/jai-dhingra)
