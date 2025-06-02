# pdf_chatbot_flask/app.py

import os
import fitz  # PyMuPDF
import openai
import faiss
import pickle
import numpy as np
from flask import Flask, request, render_template
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory store of all chunks
CHUNKS = []
VEC_INDEX = None

# PDF Processing

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_texts(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    return [record["embedding"] for record in response["data"]]

def build_vector_store(chunks):
    embeddings = embed_texts(chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def find_similar_chunks(query, index, chunks, k=5):
    query_embedding = embed_texts([query])[0]
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    return [chunks[i] for i in I[0]]

def ask_openai(question, context_chunks):
    prompt = """Answer the question based only on the following documents:

""" + "\n---\n".join(context_chunks) + f"\n\nQuestion: {question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful legal assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response["choices"][0]["message"]["content"].strip()

@app.route("/", methods=["GET", "POST"])
def index():
    global CHUNKS, VEC_INDEX
    answer = ""
    if request.method == "POST":
        if "document" in request.files:
            files = request.files.getlist("document")
            for pdf in files:
                path = os.path.join(UPLOAD_FOLDER, pdf.filename)
                pdf.save(path)
                text = extract_text_from_pdf(path)
                pdf_chunks = chunk_text(text)
                CHUNKS.extend(pdf_chunks)
            VEC_INDEX = build_vector_store(CHUNKS)

        if request.form.get("question"):
            question = request.form["question"]
            if VEC_INDEX is None or not CHUNKS:
                answer = "Please upload PDF documents first."
            else:
                relevant_chunks = find_similar_chunks(question, VEC_INDEX, CHUNKS)
                answer = ask_openai(question, relevant_chunks)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
