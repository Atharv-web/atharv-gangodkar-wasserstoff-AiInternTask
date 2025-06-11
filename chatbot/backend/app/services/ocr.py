import os
from typing import List
from PIL import Image
import pytesseract
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

UPLOAD_FOLDER = os.path.join(os.getcwd(), "data", "uploads")
rcts_model = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 200)

def handle_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    return rcts_model.split_documents(loader.load())

def handle_txt(file_path):
    loader = TextLoader(file_path)
    return rcts_model.split_documents(loader.load())

def handle_csv(file_path):
    loader = CSVLoader(file_path)
    return rcts_model.split_documents(loader.load())

def handle_image(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    doc = Document(page_content=text, metadata={"file_path": file_path})
    return rcts_model.split_documents([doc])

# Mapping of extensions to their handler functions
EXTENSION_HANDLER_MAP = {
    ".pdf": handle_pdf,
    ".txt": handle_txt,
    ".csv": handle_csv,
    ".jpg": handle_image,
    ".jpeg": handle_image,
    ".png": handle_image
}

def load_user_data(filenames: List[str]) -> List[Document]:
    for file in filenames:
        file_path = os.path.join(UPLOAD_FOLDER, file)
        ext = os.path.splitext(file)[1].lower()

        handler = EXTENSION_HANDLER_MAP.get(ext)
        if handler:
            try:
                docs = handler(file_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        else:
            print(f"Unsupported file type: {file}")

    return docs
