
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader,TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from PIL import Image
import pytesseract
import faiss
import os

embedder_model = OllamaEmbeddings(model= 'nomic-embed-text:latest')
rcts_model = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap= 200)

index = faiss.IndexFlatL2(768) # we are using nomic embed text model, hence embedding_size = 768

UPLOAD_FOLDER = "data/uploads"
DATABASE_NAME = 'data/VECTOR_STORE'

def load_data(filepaths: list[str]):
    all_documents = []
    for file_path in filepaths:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            loader = PyMuPDFLoader(file_path=file_path)
            all_documents.extend(loader.load())
        elif file_extension == ".txt":
            loader = TextLoader(file_path=file_path)
            all_documents.extend(loader.load())
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            try:
                image = Image.open(file_path)
                text_content = pytesseract.image_to_string(image)
                # Create a Langchain Document
                document = Document(page_content=text_content, metadata={"source": file_path})
                all_documents.append(document)
            except Exception as e:
                print(f"Error processing image file {file_path}: {e}")
        else:
            print(f"Unsupported file type: {file_extension} for file {file_path}")


    if not all_documents:
        return [] #returning empty list if nothing is processed.

    chunked_docs = rcts_model.split_documents(all_documents)
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