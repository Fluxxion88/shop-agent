from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from shop_agent.gemini_client import GeminiClient
from shop_agent.orchestrator import Orchestrator
from shop_agent.policy import PolicyEngine
from shop_agent.state import load_session, save_session

app = FastAPI()


def _build_orchestrator() -> Orchestrator:
    gemini = GeminiClient()
    policy_engine = PolicyEngine.from_file(Path(__file__).resolve().parents[1] / "policies.json")
    return Orchestrator(gemini, policy_engine)


@app.post("/chat")
async def chat(session_id: str = Form(...), message: str = Form(...)) -> JSONResponse:
    orchestrator = _build_orchestrator()
    state = load_session(session_id)
    response = orchestrator.handle_turn(state, message)
    save_session(state)
    return JSONResponse({"response": response, "state": state.to_json()})


@app.post("/chat_with_image")
async def chat_with_image(
    session_id: str = Form(...),
    message: str = Form(...),
    image: UploadFile = File(...),
) -> JSONResponse:
    orchestrator = _build_orchestrator()
    state = load_session(session_id)
    image_bytes = await image.read()
    response = orchestrator.handle_turn(state, message, image_bytes=image_bytes)
    save_session(state)
    return JSONResponse({"response": response, "state": state.to_json()})
