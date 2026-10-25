# AutoPDI backend
1. python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
2. export ANTHROPIC_API_KEY=your_key
3. Put OEM manuals (PDF, MD or TXT) in ./kb, or POST them to /api/ingest
4. uvicorn app:app --host 0.0.0.0 --port 8000
Endpoints: GET /api/health, POST /api/chat {message, history}, POST /api/inspect (images, part), POST /api/ingest (file)
In the AutoPDI page, open Settings and set the Backend URL to http://localhost:8000
