
from models.embeddings import DATABASE_NAME
from langchain_community.vectorstores import FAISS
from models.embeddings import EMBEDDER_MODEL
from models.llm import get_response

def retrieve_docs(query):
    global vectordatabase

    vectordatabase = FAISS.load_local(
        DATABASE_NAME,
        embeddings=EMBEDDER_MODEL,
        allow_dangerous_deserialization=True,
    )
    retriever = vectordatabase.as_retriever(
        search_kwargs={"k": 3}
    )
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs

def answer_query_chatbot(session_history):
    return get_response(session_history)