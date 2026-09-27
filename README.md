# PDF Intelligence Platform

Production-oriented FastAPI PDF summarization application with:

- Modern drag-and-drop web UI
- PDF validation
- Background processing
- OpenAI summarization engine
- Automatic extractive fallback
- Job status API
- Health endpoint
- Structured logging
- Error handling
- Section-level summaries
- Executive summary
- Copy/download results

## 1. Create virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install packages

```bash
pip install -r requirements.txt
```

## 3. Configure environment

Copy `.env.example` to `.env`.

Windows:

```powershell
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Then configure:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

If `OPENAI_API_KEY` is not configured, the application automatically uses the extractive fallback engine.

## 4. Start application

```bash
python run.py
```

