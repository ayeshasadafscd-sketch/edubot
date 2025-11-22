import os
import gradio as gr
import fitz  # PyMuPDF
import numpy as np
import faiss
import traceback

from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============================== CONFIG ===============================
GROQ_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    print("❌ WARNING: GROQ_API_KEY is missing. Add it in environment variables")

client = Groq(api_key=GROQ_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =============================== PDF READER ===============================
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# =============================== CHUNKING ===============================
def chunk_text(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# =============================== FAISS ===============================
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, chunks, index, k=3):
    q_embed = embedder.encode([query]).astype("float32")
    distances, idx = index.search(q_embed, k)
    return [chunks[i] for i in idx[0]]

# =============================== LLM FUNCTIONS ===============================
def ask_llm(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception:
        return traceback.format_exc()

def explain_topic(text):
    return ask_llm(f"Explain in simple words:\n{text}")

def summarize_text(text):
    return ask_llm(f"Summarize clearly:\n{text}")

def generate_mcqs(text):
    return ask_llm(f"Generate 5 MCQs with answers:\n{text}")

# =============================== RAG PIPELINE ===============================
def rag_answer(pdf_file, question):
    try:
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(text)
        index = build_faiss_index(chunks)
        retrieved = retrieve(question, chunks, index, k=3)
        context = "\n\n".join(retrieved)

        prompt = f"""
Use ONLY the context below to answer the question.
If the answer does not exist, say:
"The document does not contain this information."

Context:
{context}

Question:
{question}
"""

        return ask_llm(prompt)

    except Exception:
        return traceback.format_exc()

# =============================== UI ===============================
with gr.Blocks(title="EduBot+") as app:
    gr.Markdown("# 🎓 EduBot+ — AI Study Assistant")

    # Explain Tab
    with gr.Tab("Explain"):
        inp = gr.Textbox(label="Enter topic or text")
        pdf = gr.File(label="Upload PDF (optional)")
        out = gr.Textbox(label="Explanation")
        gr.Button("Explain").click(
            lambda x, f: explain_topic(extract_text_from_pdf(f) if f else x),
            [inp, pdf], out)

    # Summarize Tab
    with gr.Tab("Summarize"):
        inp = gr.Textbox(label="Enter text or PDF")
        pdf = gr.File(label="Upload PDF (optional)")
        out = gr.Textbox(label="Summary")
        gr.Button("Summarize").click(
            lambda x, f: summarize_text(extract_text_from_pdf(f) if f else x),
            [inp, pdf], out)

    # MCQs Tab
    with gr.Tab("Generate MCQs"):
        inp = gr.Textbox(label="Enter text or PDF")
        pdf = gr.File(label="Upload PDF (optional)")
        out = gr.Textbox(label="Generated MCQs")
        gr.Button("Generate MCQs").click(
            lambda x, f: generate_mcqs(extract_text_from_pdf(f) if f else x),
            [inp, pdf], out)

    # RAG Tab
    with gr.Tab("Ask PDF (RAG)"):
        pdf_file = gr.File(label="Upload PDF")
        question = gr.Textbox(label="Your Question")
        answer = gr.Textbox(label="Answer")
        gr.Button("Ask").click(rag_answer, [pdf_file, question], answer)

app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
