# Shop Agent

Policy-safe AI agent for returns, refunds, and discounts using Gemini with persistent storage.

## Requirements
- macOS with Python 3.11+
- Gemini API key in `GEMINI_API_KEY`

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## Environment
```bash
export GEMINI_API_KEY="your_key_here"
export ADMIN_PASSWORD="your_admin_password"
export CORS_ALLOWED_ORIGINS="https://your-lovable-app.com,http://localhost:3000"
export DATABASE_URL="postgresql+psycopg://user:pass@host:5432/shop_agent"
```

If `DATABASE_URL` is not set, the app uses `sqlite:///./shop_agent.db`.

## Run CLI
```bash
python -m shop_agent.cli demo-session "My furniture arrived broken"
python -m shop_agent.cli demo-session "4 days"
python -m shop_agent.cli demo-session "Not assembled"
```

Example multi-turn CLI flow with image:
```bash
python -m shop_agent.cli demo-session "My electronics arrived broken" --image ./photo.jpg
# Bot asks: "Do you have evidence of the defect (image/video or clear symptoms)?"
python -m shop_agent.cli demo-session "4 days"
python -m shop_agent.cli demo-session "Not assembled"
# Bot replies with a deterministic decision or asks for data collection.
```

## Run Server
```bash
uvicorn shop_agent.server:app --host 0.0.0.0 --port 8000
```

### Example requests
Text-only:
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-session","message":"My furniture arrived broken"}'
```

Text + image:
```bash
curl -X POST http://127.0.0.1:8000/api/chat-with-image \
  -F session_id=demo-session \
  -F message="This item arrived damaged, can I return it?" \
  -F image=@./photo.jpg
```

## Admin API
```bash
curl -H "X-Admin-Password: $ADMIN_PASSWORD" http://127.0.0.1:8000/api/admin/cases
curl -H "X-Admin-Password: $ADMIN_PASSWORD" http://127.0.0.1:8000/api/admin/cases/1
```

## Notes
- Decisions are deterministic and enforced by the decision tree in code.
- The LLM only extracts structured facts and never decides eligibility or discounts.
- The dialog manager enforces a max of 8 turns and avoids repeated questions.
