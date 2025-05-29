
# Excel Data RAG Chatbot with Ollama and FAISS

This project provides a Streamlit application that allows you to upload Excel files, extract text, generate embeddings using a HuggingFace model, store them in a FAISS vector store, and then chat with your data using the Qwen3 Large Language Model powered by Ollama.

## Features

-   **Excel Data Extraction:** Reads text content from all sheets and rows of an uploaded Excel file (`.xlsx`, `.xls`).
    
-   **HuggingFace Embeddings:** Generates vector embeddings for the extracted text using the `sentence-transformers/all-MiniLM-L6-v2` model.
    
-   **FAISS Vector Store:** Efficiently stores and retrieves embeddings using FAISS (Facebook AI Similarity Search).
    
-   **Persistent Storage:** Saves FAISS indexes locally, allowing you to load and chat with previously processed Excel data without re-embedding.
    
-   **Ollama Integration:** Connects to a local Ollama instance to leverage the Qwen3 LLM for retrieval-augmented generation (RAG).
    
-   **Intelligent Prompting:** Custom prompt engineering guides the LLM to understand the numerical and structured nature of Excel data for more accurate and concise answers.
    

## Tech Stack

-   **Frontend/Web Framework:** Streamlit
    
-   **Data Handling:** Pandas
    
-   **Deep Learning Framework:** PyTorch
    
-   **NLP/Embeddings:** HuggingFace Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
    
-   **Vector Search:** FAISS (Facebook AI Similarity Search)
    
-   **LLM Orchestration:** LangChain
    
-   **Local LLM Runtime:** Ollama
    
-   **Large Language Model:** Qwen3 (via Ollama)
    

## Requirements

Before you begin, ensure you have Python 3.8+ installed.

The Python dependencies are listed in `requirements.txt`.

## Step-by-Step Environment Setup

1.  **Clone the Repository (if applicable):**
    
    ```
    git clone <your-repository-url>
    cd <your-project-directory>
    
    
    ```
    
    (If you downloaded the script directly, navigate to its directory.)
    
2.  **Create a Virtual Environment (Recommended):**
    
    ```
    python -m venv venv
    
    
    ```
    
3.  **Activate the Virtual Environment:**
    
    -   **On Windows:**
        
        ```
        .\venv\Scripts\activate
        
        
        ```
        
    -   **On macOS/Linux:**
        
        ```
        source venv/bin/activate
        
        
        ```
        
4.  **Install Python Dependencies:**
    
    ```
    pip install -r requirements.txt
    
    
    ```
    
    _(If you don't have a `requirements.txt`, create one with the following content and then run the command above:)_
    
    ```
    streamlit
    pandas
    openpyxl
    torch
    transformers
    faiss-cpu # Use faiss-gpu if you have a compatible GPU
    numpy
    langchain-community
    langchain-core
    langchain
    
    
    ```
    
5.  **Install Ollama:** Download and install Ollama from the official website: [https://ollama.com/](https://ollama.com/ "null") Follow the installation instructions for your operating system.
    
6.  **Download the Qwen3 LLM Model via Ollama:** Open your terminal or command prompt and run:
    
    ```
    ollama run qwen3
    
    
    ```
    
    This command will download the Qwen3 model and start it. Keep Ollama running in the background while using the Streamlit app.
    
7.  **Create the `faiss_indexes` Folder:** This folder will store your generated FAISS vector stores.
    
    ```
    mkdir faiss_indexes
    
    
    ```
    

## How to Run Locally

1.  **Ensure Ollama is running** with the Qwen3 model (as per step 6 in "Step-by-Step Environment Setup").
    
2.  **Activate your virtual environment** (if not already active).
    
3.  **Navigate to the directory** containing your Streamlit application script (`app.py`).
    
4.  **Run the Streamlit application:**
    
    ```
    streamlit run app.py
    
    
    ```
    
    (Replace `app.py` with the actual name of your Python script.)
    
    Your web browser should automatically open a new tab with the Streamlit application. If not, copy the URL displayed in your terminal (e.g., `http://localhost:8501`).
    

## Usage

1.  **Upload New Excel File:**
    
    -   In the "Upload New Excel File" section, click "Choose an Excel file" and select your `.xlsx` or `.xls` document.
        
    -   The application will extract text, generate embeddings, and create a new FAISS vector store, saving it in the `faiss_indexes` directory.
        
2.  **Chat with Existing FAISS Index:**
    
    -   Once an Excel file has been processed and saved, its name will appear in the "Select a FAISS index to chat with:" dropdown.
        
    -   Select an index. The application will load the pre-computed vector store.
        
    -   Enter your questions in the text input field, and the Qwen3 LLM will provide answers based on the content of your Excel file.