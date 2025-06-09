from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from app.services.vector_db import process_and_store_documents
from app.services.query import answer_question_with_themes


app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "data", "uploads")
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'txt'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

if __name__ == '__main__':
    app.run(debug=True)
