from flask import Flask, request, render_template, jsonify
from langchain_core.messages import HumanMessage,AIMessage
from werkzeug.utils import secure_filename
import os
from services.load_process_docs import ALLOWED_EXTENSIONS
from services.StoreDocs import process_and_store_documents
from services.query import answer_query_chatbot,retrieve_docs
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
FOLDER = os.path.join(os.getcwd(),"data")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "data", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

session_history = []
flag = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    global flag # Ensure we are modifying the global flag
    files = request.files.getlist("documents")
    file_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            file_paths.append(path)

    process_and_store_documents(file_paths)
    flag = 1
    return jsonify({"message": "Documents uploaded and processed."})
    

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided."}), 400
    
    if flag == 1:
        retrieved_docs = retrieve_docs(user_message)
        session_history.append({"retrieved_context":retrieved_docs})
        return jsonify({"retrieved_docs": retrieved_docs})
    else:
        try:
            session_history.append(HumanMessage(content=user_message))
            bot_response = answer_query_chatbot(session_history)
            session_history.append(AIMessage(content=bot_response))
            return jsonify({"response": bot_response})
        
        except Exception as e:
            return jsonify({
                "response": "⚠️ Sorry, an unexpected error occurred.",
                "error": str(e)
            }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)