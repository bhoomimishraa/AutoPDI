"""AutoPDI backend: RAG (FAISS) + vision-language model + zero-shot OOD.
Run: export ANTHROPIC_API_KEY=...  &&  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import base64, glob, io, json, os
import faiss, numpy as np, torch
from anthropic import Anthropic
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

MODEL = os.getenv("VLM_MODEL", "claude-sonnet-5-5")
client = Anthropic()
app = FastAPI(title="AutoPDI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- RAG over OEM documents ----------
emb = SentenceTransformer("all-MiniLM-L6-v2")
chunks, sources = [], []
index = faiss.IndexFlatIP(emb.get_sentence_embedding_dimension())

def split(text, size=600, overlap=100):
    return [text[i:i + size] for i in range(0, max(len(text), 1), size - overlap) if text[i:i + size].strip()]

def add_text(text, src):
    parts = split(text)
    if not parts: return 0
    v = emb.encode(parts, normalize_embeddings=True)
    index.add(np.asarray(v, dtype="float32")); chunks.extend(parts); sources.extend([src] * len(parts))
    return len(parts)

def read_file(path, data=None):
    if path.lower().endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data) if data else path).pages)
    return data.decode("utf8", "ignore") if data else open(path, encoding="utf8").read()

def retrieve(q, k=4):
    if index.ntotal == 0: return []
    s, i = index.search(np.asarray(emb.encode([q], normalize_embeddings=True), dtype="float32"), min(k, index.ntotal))
    return [{"text": chunks[j], "source": sources[j], "score": float(x)} for x, j in zip(s[0], i[0]) if j >= 0]

for f in glob.glob("kb/*"): add_text(read_file(f), os.path.basename(f))

# ---------- Zero-shot OOD with CLIP ----------
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
KNOWN = ["a car body panel", "a car headlight", "a car tyre and wheel", "a car interior dashboard", "a car engine bay",
         "a car seat", "a car windscreen", "a car tail light", "a car door", "a car bumper"]
DEFECT = ["a dented car panel", "a scratched car paint", "a cracked glass", "a leaking engine fluid", "a torn car seat"]
OOD_THRESHOLD = float(os.getenv("OOD_THRESHOLD", "0.22"))

@torch.no_grad()
def ood_check(img):
    labels = KNOWN + DEFECT
    x = proc(text=labels, images=img, return_tensors="pt", padding=True)
    out = clip(**x)
    sims = torch.nn.functional.cosine_similarity(out.image_embeds, out.text_embeds)
    best = int(sims.argmax())
    return {"closest_label": labels[best], "max_similarity": float(sims.max()),
            "is_ood": bool(sims.max() < OOD_THRESHOLD), "defect_like": best >= len(KNOWN)}

# ---------- VLM ----------
def b64(img_bytes, mt): return {"type": "image", "source": {"type": "base64", "media_type": mt, "data": base64.b64encode(img_bytes).decode()}}

def ask(system, content, max_tokens=900):
    r = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": content}])
    return "".join(b.text for b in r.content if b.type == "text")

# ---------- API ----------
@app.get("/api/health")
def health(): return {"ok": True, "chunks": len(chunks)}

@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    n = add_text(read_file(file.filename, await file.read()), file.filename)
    return {"added_chunks": n}

@app.post("/api/chat")
async def chat(payload: dict):
    q = payload["message"]; hits = retrieve(q)
    ctx = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits) or "No documents found."
    system = ("You are the AutoPDI assistant for pre-delivery vehicle inspection. Answer using the OEM context. "
              "If the context lacks the answer, say so and give general guidance. Be concise and practical.")
    hist = [{"role": m["role"], "content": m["content"]} for m in payload.get("history", [])][-6:]
    r = client.messages.create(model=MODEL, max_tokens=700, system=system,
        messages=hist + [{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}])
    return {"answer": "".join(b.text for b in r.content if b.type == "text"), "sources": sorted({h["source"] for h in hits})}

@app.post("/api/inspect")
async def inspect(images: list[UploadFile] = File(...), part: str = Form("vehicle")):
    results = []
    for up in images:
        raw = await up.read(); img = Image.open(io.BytesIO(raw)).convert("RGB")
        ood = ood_check(img)
        ctx = "\n".join(h["text"] for h in retrieve(f"{part} inspection defects", 3))
        text = ask("You are a PDI inspector. Return ONLY JSON: {\"part\":str,\"defects\":[{\"type\":str,\"severity\":\"low|medium|high\",\"detail\":str}],\"verdict\":\"pass|fail|review\"}.",
                   [b64(raw, up.content_type or "image/jpeg"), {"type": "text", "text": f"Inspect this {part} photo. OEM guidance:\n{ctx}"}])
        try: finding = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except Exception: finding = {"verdict": "review", "defects": [], "raw": text}
        if ood["is_ood"]: finding["verdict"] = "review"; finding["note"] = "Unusual image, needs manual review."
        results.append({"file": up.filename, "ood": ood, "finding": finding})
    return {"results": results}
