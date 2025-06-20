# Document Q&A Chatbot

This repository contains a Document Q&A Chatbot backend built with Python and Flask. It provides API endpoints for uploading documents and asking questions about their content using modern AI and vector search technologies.

## Features

- Upload PDF, image (PNG, JPG, JPEG), or TXT files for analysis
- Ask questions about uploaded documents and receive AI-generated answers
- Supports OCR for images and advanced text embeddings
- Uses FAISS for fast vector similarity search
- Integrates with Google Generative AI and Anthropic models

## Directory Structure

```
chatbot/
  ├── my_project/           # Main application code
  ├── requirements.txt      # Python dependencies
  ├── README.md             # Project documentation
  └── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Atharv-web/atharv-gangodkar-wasserstoff-AiInternTask.git
cd atharv-gangodkar-wasserstoff-AiInternTask/chatbot
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # (On Windows: venv\Scripts\activate)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

Create a `.env` file in the `chatbot` directory with the following environment variables:

```
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
FLASK_ENV=development
FLASK_APP=app.main
SECRET_KEY=your_secret_key_here
VECTOR_DB_PATH=./data/vector_db
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=./data/uploads
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### 5. Create necessary directories

```bash
mkdir -p data/uploads
mkdir -p data/vector_db
```

### 6. Run the app

```bash
flask run
```

The server will be available at [http://localhost:5000](http://localhost:5000).

## API Endpoints

- `POST /upload`  
  Upload and process documents (form field: `documents`).  
  Supported: PDF, PNG, JPG, JPEG, TXT.

- `POST /ask`  
  Ask questions about uploaded documents (form field: `question`).  
  Returns a JSON answer.

## Deployment

You can deploy this backend to Render or any other cloud provider that supports Python web services.  
Make sure to set all environment variables in your hosting dashboard.

## Dependencies

- Flask
- FAISS
- sentence-transformers
- pytesseract
- pymupdf
- pillow
- google-generativeai
- python-dotenv
- langchain (+ community, core, ollama, google-genai, anthropic)

See `chatbot/requirements.txt` for the full list.

**Contributions are welcome!**
