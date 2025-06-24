# 🧠 NEET AI Coach

An intelligent, adaptive AI assistant designed to help students prepare for NEET (Medical Entrance Exam in India).  
This system ingests NCERT content, solves MCQs, detects trick questions, and provides personalized tutoring across Physics, Chemistry, and Biology.

---

## 🚀 Features

- **LLM-Powered**: Uses local language models (via Ollama) for explanations, problem solving, and conversation
- **Multi-Interface**: Includes both Streamlit web UI and CLI terminal interface
- **MCQ Analysis**: Parses and solves multiple-choice questions with reasoning
- **Trick Detection**: Highlights misleading or confusing question patterns
- **Personalization**: Tracks user performance, flags weak topics, adapts tutoring focus
- **Session Memory**: Retains conversation history and learning progress
- **Knowledge Ingestion**: Processes NCERT PDFs, DOCX notes, and CSV/Excel MCQs

---

## 🧹 Project Structure

```
neet_ai_coach/
├── start_coach.py               # Main launcher
├── config/config.yaml           # Configuration file
├── requirements.txt             # Python dependencies
├── data/books/                  # NCERT textbooks (PDF/DOCX)
├── data/mcqs/                   # MCQ banks (CSV, JSON)
├── core/                        # Core agent logic and plugins
├── ui/                          # Streamlit and CLI interfaces
├── memory/                      # Vector store management
├── utils/                       # Math solver, trick detector, etc.
```

---

## 🛠️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/neet-ai-coach.git
cd neet-ai-coach
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your study materials
- Place **NCERT PDFs or DOCX** into: `data/books/`
- Place **MCQs (CSV/JSON)** into: `data/mcqs/`

---

## ▶️ Running the App

### Option 1: Command Line Interface (CLI)
```bash
python start_coach.py
```

### Option 2: Web App (Streamlit)
```bash
streamlit run ui/streamlit_ui.py
```

---

## ⚙️ Configuration

Edit `config/config.yaml` to change model, paths, or behavior:
```yaml
model_name: "mistral"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
index_path: "memory/index"
session_memory: true
```

---

## 🧠 Model Information

- Language Models: Mistral / Phi (via [Ollama](https://ollama.com))
- Embedding Models: Sentence Transformers
- Vector Store: ChromaDB (or FAISS)

---

## 🧪 Roadmap

- [ ] Exam simulation mode (timed mock tests)
- [ ] MCQ generator (create new practice questions)
- [ ] Multi-user student login + analytics dashboard
- [ ] Mobile wrapper (Flutter or React Native)

---

## 👨‍🏫 Built By

You. With the occasional unsolicited genius of Claude and ChatGPT.  
Go ace NEET, or help someone else do it.
