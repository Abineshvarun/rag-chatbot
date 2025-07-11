# InsightIQ Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot that supports multi-format document upload (PDF, CSV, DOCX, PPTX, TXT, XML, and images). It utilizes local models (Ollama with LLaMA3), LangChain for document parsing, ChromaDB for vector storage, and advanced features like YOLOv8 for object detection and EasyOCR for text recognition in images.

## 🔥 Features

- 💬 Chat interface using Streamlit
- 📄 Supports PDF, CSV, DOCX, PPTX, TXT, XML, and image (JPG, PNG) inputs
- 🔍 Document search and question answering using RAG
- 🧠 Local LLM inference with Ollama (LLaMA3 or any supported model)
- 🖼️ YOLOv8 object detection for images
- 🔡 EasyOCR for extracting text from image files
- 🧠 Embeddings via `nomic-embed-text-v1`
- 📚 Chunking and vector storage with ChromaDB
- 🗂️ CrossEncoder reranking for response accuracy

## 🚀 Installation

1. Clone the repository
   
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Set up a Python virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\

3. Install dependencies

pip install -r requirements.txt

4. Install and run Ollama

Download from: https://ollama.com

ollama serve
ollama run llama3

5.Launch the Streamlit app

streamlit run app.py
