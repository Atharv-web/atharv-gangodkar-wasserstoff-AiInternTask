
import os
from PIL import Image
import pytesseract
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

UPLOAD_FOLDER = os.path.join(os.getcwd(), "data", "uploads")
RCTS_MODEL = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

ALLOWED_EXTENSIONS = {'pdf','txt','csv', 'png', 'jpg'}

def handle_pdf_data(file_path):
    loader = PyMuPDFLoader(file_path)
    split_docs = RCTS_MODEL.split_documents(loader.load())
    return split_docs

def handle_txt_data(file_path):
    loader = TextLoader(file_path)
    split_docs = RCTS_MODEL.split_documents(loader.load())
    return split_docs

def handle_csv_data(file_path):
    loader = CSVLoader(file_path)
    split_docs = RCTS_MODEL.split_documents(loader.load())
    return split_docs

def handle_image_data(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    doc = Document(page_content=text, metadata={"file_path": file_path})
    split_docs = RCTS_MODEL.split_documents([doc])
    return split_docs

def process_user_data(filenames):
    all_chunked_data = []
    for file_path in filenames:
        extension = os.path.splitext(file_path)[1].lower()
        if extension in ALLOWED_EXTENSIONS:
            try:
                if extension == ".pdf":
                    chunked_docs = handle_pdf_data(file_path)
                elif extension == ".csv":
                    chunked_docs = handle_csv_data(file_path)
                elif extension == ".txt":
                    chunked_docs = handle_txt_data(file_path)
                else:
                    chunked_docs = handle_image_data(file_path)
                all_chunked_data.extend(chunked_docs)
            except Exception as e:
                return f"There was an error in processing the docs. ERROR Occured: {e}"
            
        else:
            raise ValueError(f"File extension {extension} is not allowed")
    return all_chunked_data