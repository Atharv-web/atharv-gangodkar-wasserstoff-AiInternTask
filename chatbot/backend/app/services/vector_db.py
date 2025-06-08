from app.models.embeddings import embed_and_index,load_pdf_data

def process_and_store_documents(filepaths):
    loaded_docs = load_pdf_data(filepaths)
    embed_and_index(loaded_docs)