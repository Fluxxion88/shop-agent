# Shop Agent

Policy-safe AI agent for returns, refunds, and discounts using Gemini.

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
```

## Run CLI
```bash
python -m shop_agent.cli demo-session "I want a refund for these headphones"
python -m shop_agent.cli demo-session "Discount request for this chair" --image ./chair.jpg
```

## Run Server
```bash
uvicorn shop_agent.server:app --reload
```

### Example requests
Text-only:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -F session_id=demo-session \
  -F message="I want a 25% discount on my phone"
```

Text + image:
```bash
curl -X POST http://127.0.0.1:8000/chat_with_image \
  -F session_id=demo-session \
  -F message="This item arrived damaged, can I return it?" \
  -F image=@./photo.jpg
```

## Notes
- Policies are deterministic and enforced by `policies.json`.
- If the image classifier returns confidence below 0.70 or needs clarification, the agent asks follow-up questions.
