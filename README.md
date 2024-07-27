# Build a RAG based chat application
This project is about building a RAG based chat application using the following components.
- Chroma vector store: To persist the embedded document contents.
- Embedding and LLM models hosted on watsonx.AI: To generate the document embedding and generate text response respectively.
- Jupyter Notebook: Contains steps to download the files, create embeddings and persist them to local Chroma DB.
- Streamlit: Python library to build a chat application.

## Pre-requisites
* Install Python 3.11
    Ref. using pyenv
     - Windows: https://github.com/pyenv-win/pyenv-win
     - Mac: https://github.com/pyenv-win/pyenv
     
* Install below Python packages

    pip install "langchain==0.1.10"

    pip install "ibm-watsonx-ai>=0.2.6"

    pip install -U langchain_ibm

    pip install wget

    pip install sentence-transformers

    pip install "chromadb==0.3.26"

    pip install "chromadb-client"

    pip install "pydantic==1.10.0"

    pip install "sqlalchemy==2.0.1"

    pip install "pypdf"

    pip install "jupyter"

    pip install "streamlit"


## How to run this application.
1. Download the documents, generate embedding and persist it to the local vector store. Refer to the details in section - `Steps to start the Jupyter notebook to load documents to vector store`
2. Start the chat application to chat with the LLM. Refer to the detail in section - `Steps to start the streamlit web application`


## Steps to start the Jupyter notebook to load documents to vector store
1. Go to the directory - `[Project Root]/notebooks` in terminal / command line
2. Set the python virtual environment using the command below in the terminal / command line.

    `pyenv local 3.11.0`
3. Start the notebook service using the command below in the terminal / command line.

    `jupyter notebook`
   
   This will open a tab in the default browser that list the files inside the `[Project Root]/notebooks`

4. Click on the file `Local-Basic-RAG.ipynb` to open it.


## Steps to start the streamlit web application
1. Go to the directory - `[Project Root]/basic-rag` in terminal / command line
2. Start the streamlit service using the below command.

    `streamlit run basic-rag-app.py`
