from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os

vecdb = None
EMBEDDER_MODEL = OllamaEmbeddings(model='nomic-embed-text:latest')
DATABASE_NAME = os.path.join(os.getcwd(), "data", "VECTOR_STORE")

def embed_and_index(chunked_docs): # loaded_docs = chunked data which is processed
    global vecdb
    dimension = 768  # nomic-embed-text embedding dimension
    index = faiss.IndexFlatL2(dimension)
    
    # Initialize the vector store
    vecdb = FAISS(
        embedding_function=EMBEDDER_MODEL,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )
    # Add chunked_documents to the vector store
    vecdb.save_local(DATABASE_NAME)
    vecdb.add_documents(chunked_docs)