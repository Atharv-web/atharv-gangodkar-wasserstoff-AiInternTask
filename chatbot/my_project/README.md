# Document Q&A Chatbot Backend

This is the backend application for the Document Q&A Chatbot. It provides API endpoints for document processing and question answering.

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# API Keys
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Application Settings
FLASK_ENV=development
FLASK_APP=app.main
SECRET_KEY=your_secret_key_here

# Database Settings
VECTOR_DB_PATH=./data/vector_db

# File Upload Settings
MAX_CONTENT_LENGTH=16777216  # 16MB in bytes
UPLOAD_FOLDER=./data/uploads

# Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Default sentence transformer model
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create required directories:
```bash
mkdir -p data/uploads
mkdir -p data/vector_db
```

4. Start the development server:
```bash
flask run
```

The server will be available at `http://localhost:5000`

## API Endpoints

- POST `/upload` - Upload and process documents
  - Accepts multipart/form-data with 'documents' field
  - Supports PDF, PNG, JPG, JPEG, and TXT files

- POST `/ask` - Ask questions about uploaded documents
  - Accepts form data with 'question' field
  - Returns JSON response with answer

## Dependencies

- Flask - Web framework
- FAISS - Vector similarity search
- Sentence Transformers - Text embeddings
- PyTesseract - OCR for images
- PyMuPDF - PDF processing
- Google Generative AI - Language model
- LangChain - AI framework 