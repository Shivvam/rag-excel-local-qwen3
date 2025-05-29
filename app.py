import streamlit as st
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import pickle
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.documents import Document # Import Document for LangChain
from langchain.prompts import PromptTemplate 

# --- Configuration ---
FAISS_INDEX_DIR = "faiss_indexes" # Directory to store FAISS indexes
QWEN3_MODEL = "qwen3:latest"

# Ensure the FAISS index directory exists
if not os.path.exists(FAISS_INDEX_DIR):
    os.makedirs(FAISS_INDEX_DIR)

# --- Initialize HuggingFace model and tokenizer ---
@st.cache_resource
def load_hf_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # LangChain's HuggingFaceEmbeddings handles the model loading internally
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model, hf_embeddings

# --- Function to extract text from Excel ---
def extract_excel_text(file_path):
    excel_data = pd.read_excel(file_path, sheet_name=None)
    text_chunks = []
    for sheet_name, df in excel_data.items():
        for _, row in df.iterrows():
            # Combine all cell values in a row into a single string
            text = " ".join(str(cell) for cell in row.values if pd.notna(cell))
            if text.strip(): # Add only non-empty strings
                text_chunks.append(text)
    return text_chunks

# --- Function to generate embeddings (only for initial creation) ---
def generate_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# --- Function to create and save LangChain FAISS vector store ---
def create_and_save_langchain_faiss_store(text_chunks, hf_embeddings, index_name):
    # Create LangChain Document objects from text chunks
    documents = [Document(page_content=text) for text in text_chunks]

    # Create LangChain FAISS vector store from documents and embeddings
    langchain_faiss_db = FAISS.from_documents(documents, hf_embeddings)

    # Save the LangChain FAISS vector store to a directory
    save_path = os.path.join(FAISS_INDEX_DIR, index_name)
    langchain_faiss_db.save_local(save_path)
    
    return save_path

# --- Main Streamlit App ---
st.title("Chat With Excel File, 100% Local")
st.write("Upload an Excel file to extract text, generate embeddings, and store in FAISS. You can also chat with existing Excel File")

tokenizer, model, hf_embeddings = load_hf_model()

# --- Upload New Excel File ---
st.header("1. Upload New Excel File")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    with st.spinner("Processing Excel file and creating vector store..."):
        # Extract text from Excel
        text_chunks = extract_excel_text(uploaded_file)
        st.info(f"Extracted **{len(text_chunks)}** text chunks from the Excel file.")

        # Use file name as index name (remove extension)
        index_name = os.path.splitext(uploaded_file.name)[0]
        
        # Create and save LangChain FAISS vector store
        try:
            save_path = create_and_save_langchain_faiss_store(text_chunks, hf_embeddings, index_name)
            st.success(f"FAISS vector store '{index_name}' created and saved to '{save_path}' successfully!")
        except Exception as e:
            st.error(f"Error creating/saving FAISS vector store: {e}")

st.markdown("---")

## 2. Chat with Existing FAISS Index
st.header("2. Chat with Existing File")

available_indexes = [name for name in os.listdir(FAISS_INDEX_DIR) if os.path.isdir(os.path.join(FAISS_INDEX_DIR, name))]
selected_index_name = st.selectbox("Select a FAISS index to chat with:", [""] + available_indexes)

if selected_index_name:
    load_path = os.path.join(FAISS_INDEX_DIR, selected_index_name)

    with st.spinner(f"Loading FAISS index '{selected_index_name}' for chatting..."):
        try:
            langchain_faiss_db = FAISS.load_local(load_path, hf_embeddings, allow_dangerous_deserialization=True)
            st.success(f"Successfully loaded FAISS index '{selected_index_name}' for chatting.")

            try:
                llm = Ollama(model=QWEN3_MODEL)
                st.write(f"**Initialized LLM: {QWEN3_MODEL}**")

                # --- Custom Prompt Template ---
                # Define the custom prompt template
                # This template guides the LLM to understand the data type and desired response style.
                custom_prompt_template = """You are an AI assistant specialized in analyzing data from Excel files.
                The information provided below is extracted from an Excel spreadsheet, which means it may contain structured data, financial figures, statistical information, or other numerical data.
                When answering questions, focus on providing accurate, concise, and straightforward answers directly related to the numerical or factual content.
                Do not include any preambles like 'As an AI language model...', 'I think...', or any internal thought processes.
                Always answer directly. If the information is not available in the provided context, state that clearly and concisely.
                Do not include the reasoning part (i.e. <think></think> ) in answer just provide the number/text of answer, as concise as possible

                Context:
                {context}
                
                Question:
                
                {question}
                
                Straightforward Answer:"""
                
                # Create a PromptTemplate object
                CUSTOM_PROMPT = PromptTemplate(
                    template=custom_prompt_template,
                    input_variables=["context", "question"]
                )

                # Create RetrievalQA chain with custom prompt
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=langchain_faiss_db.as_retriever(),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": CUSTOM_PROMPT} # Pass the custom prompt here
                )

                # Chat input
                user_query = st.text_input("Ask a question about the loaded Excel data:")
                if user_query:
                    with st.spinner("Getting answer..."):
                        response = qa_chain({"query": user_query})
                        st.write("---")
                        st.write("### Answer:")
                        st.write(response["result"])
                        
                         
                            

            except Exception as e:
                st.error(f"Error initializing or using Ollama: {e}")
                st.warning("Please ensure Ollama is running and the 'qwen3' model is downloaded. You can run `ollama run qwen3` in your terminal.")

        except Exception as e:
            st.error(f"Error loading FAISS index from '{load_path}': {e}")
            st.warning("This could be due to a corrupted index or an index saved with an older method. Try re-uploading the Excel file to create a new index.")

 