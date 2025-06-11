
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os

embedder_model = OllamaEmbeddings(model= 'nomic-embed-text:latest')
rcts_model = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap= 200)

index = faiss.IndexFlatL2(768) # we are using nomic embed text model, hence embedding_size = 768

DATABASE_NAME = os.path.join(os.getcwd(),"data","VECTOR_STORE")

# def load_data(filenames):
#     for file in filenames:
#         if file.endswith(".pdf"):
#             file_path = os.path.join(UPLOAD_FOLDER,file)
#             pdf_loader = PyMuPDFLoader(file_path=file_path)
#             pdf_docs = pdf_loader.load()
#             chunked_pdf_docs = rcts_model.split_documents(pdf_docs)

#         if file.endswith(".txt"):
#             file_path = os.path.join(UPLOAD_FOLDER,file)
#             text_loader = TextLoader(file_path=file_path)
#             text_docs = text_loader.load()
#             chunked_text_docs = rcts_model.split_documents(text_docs)

#         if file.endswith('.csv'):
#             file_path = os.path.join(UPLOAD_FOLDER,file)
#             csv_loader = CSVLoader(file_path)
#             csv_loaded_docs = csv_loader.load()
#             chunked_csv_docs = rcts_model.split_documents(csv_loaded_docs)

#         else:
#             if file.lower().endswith((".jpg", ".png", ".jpeg")):
#                 file_path = os.path.join(UPLOAD_FOLDER,file)
#                 img = Image.open(file_path)
#                 text += pytesseract.image_to_string(img)

#                 img_text = Document(page_content=text,metadata = {"file_path":file_path})

#                 chunked_img_docs = rcts_model.split_documents(img_text)
#                 return chunked_img_docs


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
    
    retreived_docs = vecdb.similarity_search(query)
    return retreived_docs