from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings
import chromadb
from chromadb import PersistentClient
import uuid
import numpy as np

app = FastAPI()

# Chroma client init
chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="notes")

# SBERT model
model = SentenceTransformer("cl-nagoya/ruri-small-v2", trust_remote_code=True)

@app.get("/", response_class=HTMLResponse)
async def home():
    return RedirectResponse(url="/new")

@app.get("/new", response_class=HTMLResponse)
async def new_note():
    return """<form method="post" action="/save">
                <textarea name="text" rows="10" cols="50"></textarea><br>
                <input type="submit" value="保存">
              </form>
              <a href='/search'>検索へ</a>"""

@app.post("/save", response_class=HTMLResponse)
async def save_note(text: str = Form(...)):
    embedding = model.encode(text, normalize_embeddings=True).tolist()
    note_id = str(uuid.uuid4())
    collection.add(documents=[text], embeddings=[embedding], ids=[note_id])
    return RedirectResponse(url="/new", status_code=303)

@app.get("/search", response_class=HTMLResponse)
async def search_form():
    return """<form method="post" action="/search">
                <input name="query" type="text">
                <input type="submit" value="検索">
              </form>
              <a href='/new'>新規メモ作成へ</a>"""

@app.post("/search", response_class=HTMLResponse)
async def search_results(query: str = Form(...)):
    embedding = model.encode(query, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=10,
        include=["documents", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]

    html = "<h2>検索結果（類似度 ≥ 0.6）</h2><ul>"
    for doc, dist in zip(docs, dists):
        similarity = 1 - dist  # Chromaは距離を返す
        if similarity >= 0.6:
            html += f"<li>{doc} <small>(類似度: {similarity:.3f})</small></li>"
    html += "</ul><a href='/search'>再検索</a> | <a href='/new'>新規メモ</a>"

    return html
