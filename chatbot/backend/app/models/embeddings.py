
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader,TextLoader,CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os

from app.services.ocr import extract_text_from_pics

embedder_model = OllamaEmbeddings(model= 'nomic-embed-text:latest')
rcts_model = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap= 200)

index = faiss.IndexFlatL2(768) # we are using nomic embed text model, hence embedding_size = 768

UPLOAD_FOLDER = "data/uploads"
DATABASE_NAME = 'data/VECTOR_STORE'

def load_data(filenames):
    for file in filenames:
        if file.endswith(".pdf"):
            file_path = os.path.join(UPLOAD_FOLDER,file)
            pdf_loader = PyMuPDFLoader(file_path=file_path)
            pdf_docs = pdf_loader.load()
            chunked_pdf_docs = rcts_model.split_documents(pdf_docs)

        if file.endswith(".txt"):
            file_path = os.path.join(UPLOAD_FOLDER,file)
            text_loader = TextLoader(file_path=file_path)
            text_docs = text_loader.load()
            chunked_text_docs = rcts_model.split_documents(text_docs)

        if file.endswith('.csv'):
            file_path = os.path.join(UPLOAD_FOLDER,file)
            csv_loader = CSVLoader(file_path)
            csv_loaded_docs = csv_loader.load()
            chunked_csv_docs = rcts_model.split_documents(csv_loaded_docs)

        else:
            chunked_img_docs = extract_text_from_pics(UPLOAD_FOLDER)

    chunked_docs = chunked_pdf_docs + chunked_text_docs + chunked_csv_docs + chunked_img_docs
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

def semantic_search(query):
    vecdb = FAISS.load_local(
        DATABASE_NAME,
        embeddings = embedder_model,
        allow_dangerous_deserialization = True
    )

    retriever = vecdb.as_retriever(k=3)
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs