
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os

embedder_model = OllamaEmbeddings(model= 'nomic-embed-text:latest')
rcts_model = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap= 200)

index = faiss.IndexFlatL2(768) # we are using nomic embed text model, hence embedding_size = 768

UPLOAD_FOLDER = "data/uploads"
DATABASE_NAME = 'data/VECTOR_STORE'

def load_pdf_data(filepath):
    for file in filepath:
        file_path = os.path.join(UPLOAD_FOLDER,file)
        pdf_loader = PyMuPDFLoader(file_path=file_path)
        pdf_loaded_docs = pdf_loader.load()
        chunked_docs = rcts_model.split_documents(pdf_loaded_docs)

    return chunked_docs

def embed_and_index(loaded_docs):
    vector_store = FAISS(
        embedding_function=embedder_model,
        index=index,
        index_to_docstore_id= {},
        docstore=InMemoryDocstore(),
    )

    vector_store.add_documents(loaded_docs)
    vector_store.save_local(DATABASE_NAME)

def semantic_search(query,k=1):
    vecdb = FAISS.load_local(
        DATABASE_NAME,
        embeddings = embedder_model,
        allow_dangerous_deserialization = True
    )

    retriever = vecdb.as_retriever(k=1)
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs