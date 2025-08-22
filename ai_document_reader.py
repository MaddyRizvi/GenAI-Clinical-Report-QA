import streamlit as st
import faiss
import numpy as np
import pypdf
from langchain_ollama import OllamaLLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import document

# Load AI Model
llm = OllamaLLM(model = "mistral")

# load hugging face embeddings
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS Vector Database
index = faiss.IndexFlatL2(384)
vector_store = {}           # initializing empty dictionary to store meta data and text chunks

# Function to extract text from pdf
def extract_text_from_pdf(uploaded_file):
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to store text in FAISS
def store_in_faiss(text, filename):
    global index, vector_store
    st.write(f"📥 Storing document '{filename}' in FAISS....")

    # Split texts into chunks
    splitter = CharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    texts = splitter.split_text(text)

    # convert text into embeddings
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors, dtype = np.float32)

    # store in FAISS
    index.add(vectors)
    vector_store[len(vector_store)] = (filename, texts)

    return "✅ Data stored successfully!"


# function to retrieve chunks and Answer questions 
def retrieve_and_answer(query):
    global index, vector_store

    # convert query into embeddings
    query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)

    # search FAISS
    D, I = index.search(query_vector, k =2)   # Retrive top 2 similar chunks

    content = ""
    for idx in I[0]:
        if idx in vector_store:
            content +=  " ".join(vector_store[idx][1]) + "\n\n"

    if not content:
        return " No relevant data found in stored documents"
    
    # Ask AI to generate an answer
    return llm.invoke(f"based upon the following document content, answer the question:\n\n{content}\n\n Question: {query}")

# Streamlit Web UI
st.title(" AI Document Reader & Q&A Bot")
st.write(" Upload a PDF and ask questions based on its content")

# file uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF Document", type = ["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    store_message = store_in_faiss(text, uploaded_file.name)
    st.write(store_message)

# User input for Q&A
query = st.text_input(" As a question based upon the uploaded document:")
if query:
    answer = retrieve_and_answer(query)
    st.subheader(" AI Answer")
    st.write(answer)
