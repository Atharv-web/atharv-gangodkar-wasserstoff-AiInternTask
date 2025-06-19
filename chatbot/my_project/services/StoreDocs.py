from services.load_process_docs import process_user_data
from models.embeddings import embed_and_index

def process_and_store_documents(filepaths):
    chunked_docs = process_user_data(filepaths)
    embed_and_index(chunked_docs)
    