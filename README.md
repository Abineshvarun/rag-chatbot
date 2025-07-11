<h1 align="left">💬 InsightIQ RAG Chatbot</h1>

<p align="left">
A powerful multi-format document assistant built using LangChain, ChromaDB, Ollama (LLaMA3), and Streamlit.
InsightIQ supports PDF, CSV, DOCX, PPTX, TXT, XML, XLSX, JPG files with YOLOv8 + OCR for image understanding.
It performs semantic search with re-ranking using CrossEncoder, and answers questions in real-time using a local LLM.

🚀 Features
🧠 Local LLM inference (via Ollama)

📄 Multi-format support: PDF, DOCX, PPTX, CSV, XML, XLSX, TXT, JPG

🧾 OCR-based image text extraction (EasyOCR)

🦾 Object detection with YOLOv8

🔍 Vector store powered by ChromaDB

🧩 Embeddings with nomic-embed-text

📈 Re-ranking with CrossEncoder (cross-encoder/ms-marco-MiniLM-L-6-v2)

💡 Chat history + conversational memory

🖥️ Streamlit frontend

📦 Requirements
Python 3.10+
Ollama
Git
Virtualenv (optional but recommended)

🛠️ Installation

# 1. Clone the repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama
ollama run llama3

# (Optional) Pull additional models
ollama pull llama3
