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

Optional Amazon PA-API credentials for price lookup:
```bash
export AMAZON_PAAPI_ACCESS_KEY="your_access_key"
export AMAZON_PAAPI_SECRET_KEY="your_secret_key"
export AMAZON_PAAPI_PARTNER_TAG="your_partner_tag"
export AMAZON_PAAPI_HOST="webservices.amazon.com"
export AMAZON_PAAPI_REGION="us-east-1"
```

## Run CLI
```bash
python -m shop_agent.cli demo-session "I want a refund for these headphones"
python -m shop_agent.cli demo-session "Discount request for this chair" --image ./chair.jpg
python -m shop_agent.cli demo-session "Unopened, bought 4 days ago"
```

Example multi-turn CLI flow with image:
```bash
python -m shop_agent.cli demo-session "I want a discount for this item" --image ./photo.jpg
# Bot asks: "How many days ago did you buy it?"
python -m shop_agent.cli demo-session "4 days"
# Bot asks: "Do you have the Amazon link or ASIN? If not, what was the purchase price?"
python -m shop_agent.cli demo-session "ASIN B000123456"
# Bot replies with a policy-capped discount decision.
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
- Price lookup uses Amazon PA-API when credentials are set; otherwise the agent asks for purchase price.
