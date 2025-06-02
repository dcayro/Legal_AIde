import os
import fitz  # PyMuPDF
import openai
import faiss
import pickle
import numpy as np

from flask import Flask, request, render_template
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Utilities
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def embed_texts(texts):
    response = openai.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [np.array(d.embedding, dtype=np.float32) for d in response.data]

def build_vector_store(chunks):
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, chunks

def query_vector_store(index, chunks, query, k=3):
    query_vec = embed_texts([query])[0].reshape(1, -1)
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

def ask_chatgpt(context, question):
    prompt = f"Use the following legal context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        uploaded_file = request.files["document"]
        question = request.form.get("question")

        if uploaded_file.filename != "" and question:
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(filepath)

            text = extract_text_from_pdf(filepath)
            chunks = split_text(text)
            index, chunk_list = build_vector_store(chunks)
            context = "\n\n".join(query_vector_store(index, chunk_list, question))
            answer = ask_chatgpt(context, question)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)