from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os

vecdb = None
embedder_model = OllamaEmbeddings(model='nomic-embed-text:latest')
rcts_model = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

DATABASE_NAME = os.path.join(os.getcwd(), "data", "VECTOR_STORE")

def embed_and_index(loaded_docs):
    global vecdb
    
    # Create a new FAISS index
    dimension = 768  # nomic-embed-text embedding dimension
    index = faiss.IndexFlatL2(dimension)
    
    # Initialize the vector store
    vecdb = FAISS(
        embedding_function=embedder_model,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )
    
    vecdb.save_local(DATABASE_NAME)
    # Add documents to the vector store
    vecdb.add_documents(loaded_docs)

def semantic_search(query):
    global vecdb
    
    try:
        if vecdb is None:
            if not os.path.exists(DATABASE_NAME):
                raise FileNotFoundError(f"Vector store not found at {DATABASE_NAME}")
                
            vecdb = FAISS.load_local(
                DATABASE_NAME,
                embeddings=embedder_model,
                allow_dangerous_deserialization=True
            )
        
        retriever = vecdb.as_retriever(
            search_kwargs={"k": 3}
        )
        retrieved_docs = retriever.invoke(query)
        return retrieved_docs
        
    except Exception as e:
        print(f"Error in semantic search: {str(e)}")
        return []