# AutoPDI
- frontend/index.html : the web app (open in a browser, or serve with `python -m http.server 5500` inside frontend)
- backend/ : FastAPI server (RAG + FAISS, vision-language model, zero-shot OOD). See backend/README.md

Quick start
1. cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
2. export ANTHROPIC_API_KEY=your_key
3. uvicorn app:app --host 0.0.0.0 --port 8000
4. Open frontend/index.html, go to Settings, set Backend URL to http://localhost:8000

Note: without a backend the page's AI assistant and photo analysis only work inside claude.ai.
