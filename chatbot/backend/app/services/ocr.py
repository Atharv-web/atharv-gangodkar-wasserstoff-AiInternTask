# import pytesseract
# import fitz  # PyMuPDF
# from PIL import Image
# import os
# from app.models.embeddings import embed_and_index

# def extract_text_from_file(filepath):
#     text = ""
#     if filepath.lower().endswith(".pdf"):
#         pdf_doc = fitz.open(filepath)
#         for page in pdf_doc:
#             text += page.get_text()
#             # fallback to OCR if empty
#             if not page.get_text(strip=True):
#                 pix = page.get_pixmap()
#                 img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                 text += pytesseract.image_to_string(img)
#     elif filepath.lower().endswith((".jpg", ".png", ".jpeg")):
#         img = Image.open(filepath)
#         text += pytesseract.image_to_string(img)
#     elif filepath.lower().endswith(".txt"):
#         with open(filepath, "r", encoding="utf-8") as f:
#             text += f.read()
#     return {"file_path": filepath, "content": text}

# def process_and_store_documents(filepaths):
#     docs = [extract_text_from_file(path) for path in filepaths]
#     embed_and_index(docs)

# ========================================================



