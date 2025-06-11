import pytesseract
from PIL import Image
import os
from langchain_core.documents import Document
pytesseract.pytesseract.run_tesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter

rcts = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap= 200)

def extract_text_from_pics(filepath):
    text = ""
    for file in filepath:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            file_path = os.path.join(filepath,file)
            img = Image.open(file_path)
            text += pytesseract.image_to_string(img)

            img_text = Document(page_content=text,metadata = {"file_path":file_path})

    chunked_img_docs = rcts.split_documents(img_text)
    return chunked_img_docs
