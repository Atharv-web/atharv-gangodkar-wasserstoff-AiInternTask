from flask import Flask, request, render_template, jsonify
from langchain_core.messages import HumanMessage,AIMessage
from werkzeug.utils import secure_filename
import os
from services.vector_db import process_and_store_documents
from services.query import answer_question_with_themes,chatbot
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
FOLDER = os.path.join(os.getcwd(),"data")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "data", "uploads")
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

session_history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    files = request.files.getlist("documents")
    file_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            file_paths.append(path)

    process_and_store_documents(file_paths)
    return jsonify({"message": "Documents uploaded and processed."})

@app.route("/ask", methods=["POST"])
def ask_question():
    user_question = request.form.get("question")
    if not user_question:
        return jsonify({"error": "No question provided."}), 400

    result = answer_question_with_themes(user_question)
    return jsonify(result)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided."}), 400
    
    session_history.append(HumanMessage(content=user_message))

    try:
        bot_response = chatbot(session_history)
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