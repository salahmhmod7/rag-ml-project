# 🤖 RAG ML Project

An end-to-end **Retrieval-Augmented Generation (RAG)** application for querying and extracting insights from specialized **AI, Machine Learning, and Data Science PDF documents**.

The project combines document processing, semantic search, vector databases, and Large Language Models (LLMs) to provide context-aware answers based on the uploaded knowledge base.

---

## 📖 Description

**RAG ML Project** allows users to interact with specialized machine learning literature through a conversational interface.

The system processes PDF documents by:

1. Loading PDF documents.
2. Splitting them into smaller chunks.
3. Generating vector embeddings.
4. Storing embeddings in a persistent vector database.
5. Retrieving the most relevant chunks for a user query.
6. Passing the retrieved context to an LLM.
7. Generating an answer grounded in the retrieved documents.

The knowledge base can contain resources covering topics such as:

* Machine Learning
* Deep Learning
* Data Science
* Data Analysis
* Artificial Intelligence

---

## ✨ Features

### 📚 PDF Processing

* Automatically loads PDF documents.
* Splits documents into manageable chunks.
* Prepares documents for semantic retrieval.

### 🔎 Semantic Search

* Generates embeddings for document chunks.
* Uses **ChromaDB** as a persistent vector database.
* Retrieves relevant information based on semantic similarity.

### 🤖 Retrieval-Augmented Generation

* Combines vector retrieval with an LLM.
* Provides answers based on retrieved document context.
* Reduces reliance on the LLM's internal knowledge.

### 💬 Interactive Chat Interface

* Built with **Streamlit**.
* Provides a simple conversational interface.
* Allows users to ask questions about the indexed documents.

### 🐳 Docker Support

* Includes a Dockerfile for containerized deployment.
* Provides a consistent environment across systems.

### 🔐 Environment Management

* Uses `.env` for API key management.
* Keeps sensitive credentials outside the source code.
* Includes dependency management through `requirements.txt`.

---

## 🏗️ RAG Architecture

```text
                ┌──────────────────┐
                │   PDF Documents  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  PDF Processing  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Text Chunking   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    Embeddings    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    ChromaDB      │
                │ Vector Database  │
                └────────┬─────────┘
                         │
                  User Question
                         │
                         ▼
                ┌──────────────────┐
                │ Semantic Search  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Relevant Context │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │       LLM        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │      Answer      │
                └──────────────────┘
```

---

## 🛠️ Tech Stack

| Category         | Technology                   |
| ---------------- | ---------------------------- |
| Language         | Python 3.11+                 |
| Frontend         | Streamlit                    |
| Vector Database  | ChromaDB                     |
| LLM Framework    | LangChain                    |
| LLM              | OpenAI / Compatible LLM APIs |
| Embeddings       | Embedding Model              |
| Containerization | Docker                       |
| Version Control  | Git & GitHub                 |
| Configuration    | `.env`                       |

---

## 📂 Project Structure

```text
rag-ml-project/
│
├── app.py                 # Streamlit user interface
├── main.py                # RAG pipeline and document processing
│
├── chroma_db/             # Persistent ChromaDB storage
│
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
│
├── Dockerfile             # Docker configuration
├── .env                   # Environment variables (not committed)
├── .gitignore             # Git ignore rules
│
└── *.pdf                  # Source AI/ML documents
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/salahmhmod7/rag-ml-project.git

cd rag-ml-project
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv .venv

source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Add any additional environment variables required by your selected LLM or embedding provider.

> **Important:** Never commit your `.env` file or expose your API keys publicly.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will then be available through the Streamlit interface.

---

## 🐳 Running with Docker

### Build the Docker Image

```bash
docker build -t rag-ml-app .
```

### Run the Container

```bash
docker run -p 8501:8501 rag-ml-app
```

Then open the Streamlit application in your browser.

---

## 🔄 RAG Pipeline

The application follows a typical Retrieval-Augmented Generation workflow:

```text
PDFs
  ↓
Document Loading
  ↓
Text Splitting
  ↓
Embedding Generation
  ↓
ChromaDB
  ↓
User Query
  ↓
Query Embedding
  ↓
Similarity Search
  ↓
Relevant Documents
  ↓
Context + Query
  ↓
LLM
  ↓
Generated Answer
```

---

## 🎯 Use Cases

This project can be used for:

* 📖 Asking questions about ML and AI books.
* 🧠 Learning concepts from technical documentation.
* 🔍 Searching large collections of technical PDFs.
* 📚 Building a personal AI-powered knowledge base.
* 🤖 Experimenting with Retrieval-Augmented Generation.
* 🧪 Learning vector databases and semantic retrieval.

---

## 🔮 Future Improvements

Potential improvements include:

* [ ] Streaming LLM responses.
* [ ] Source citation for retrieved documents.
* [ ] Metadata filtering.
* [ ] Hybrid search.
* [ ] Reranking retrieved documents.
* [ ] Conversation memory.
* [ ] Query rewriting.
* [ ] Multi-document management.
* [ ] Improved chunking strategies.
* [ ] Evaluation with RAG-specific metrics.
* [ ] Authentication and user management.
* [ ] Cloud deployment.
* [ ] Observability and monitoring.

---

## 👨‍💻 Author

**Salah Mahmoud**

* GitHub: [@salahmhmod7](https://github.com/salahmhmod7)
* Email: [flaysefeco.1@gmail.com](mailto:flaysefeco.1@gmail.com)

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub!

---

## 📄 License

This project is intended for educational and experimental purposes.
