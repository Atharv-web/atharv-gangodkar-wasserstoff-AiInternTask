from app.models.embeddings import embed_and_index
from app.services.ocr import load_user_data

def process_and_store_documents(filepaths):
    loaded_docs = load_user_data(filepaths)
    embed_and_index(loaded_docs)