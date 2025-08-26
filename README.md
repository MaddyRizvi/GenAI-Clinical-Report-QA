# AI-PDF-Reader-RAG

An AI-powered PDF document reader and Q&A bot using Retrieval-Augmented
Generation (RAG) with FAISS and HuggingFace embeddings.

## 📌 Features

-   Upload PDF documents via Streamlit UI
-   Extract text using `pypdf`
-   Chunk and embed text with HuggingFace embeddings
    (`all-MiniLM-L6-v2`)
-   Store and search embeddings with **FAISS**
-   Retrieve relevant chunks and generate answers with **LLM (Ollama +
    Mistral)**
-   Implements a basic **RAG (Retrieval-Augmented Generation)** pipeline

## 🚀 Installation

1.  Clone this repository:

    ``` bash
    git clone https://github.com/MaddyRizvi/GenAI-Clinical-Report-QA.git
    cd GenAI-Clinical-Report-QA
    ```

2.  Create a virtual environment and install dependencies:

    ``` bash
    conda create -n pdf-rag python=3.10 -y
    conda activate pdf-rag
    pip install -r requirements.txt
    ```

3.  Run the Streamlit app:

    ``` bash
    streamlit run ai_document_reader.py
    ```

## 🛠 Requirements

-   Python 3.9+
-   Streamlit
-   FAISS
-   pypdf
-   numpy
-   langchain
-   langchain-community
-   langchain-text-splitters
-   HuggingFace Embeddings
-   Ollama (for local LLM inference)

Install dependencies with:

``` bash
pip install streamlit faiss-cpu numpy pypdf langchain langchain-community langchain-text-splitters sentence-transformers
```

## 📂 Project Structure

    AI-PDF-Reader-RAG/
    │── ai_document_reader.py     # Main Streamlit app
    │── README.md                 # Project documentation
    │── CONTRIBUTING.md           # Contribution guidelines
    │── requirements.txt          # Python dependencies

## 🎯 Usage

1.  Start the app with Streamlit.
2.  Upload a PDF file.
3.  Ask questions in the text box.
4.  The AI will retrieve relevant chunks and generate an answer.

## 🤝 Contributing

Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file before
submitting issues or pull requests.

## 📜 License

This project is licensed under the MIT License.
