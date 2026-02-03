from __future__ import annotations

import os
from datetime import datetime

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

from shop_agent.db import Attachment, Case, Message, get_session, init_db
from shop_agent.gemini_client import GeminiClient
from shop_agent.orchestrator import DialogManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


def _build_orchestrator() -> DialogManager:
    gemini = GeminiClient()
    return DialogManager(gemini)


def _admin_auth(x_admin_password: str = Header(default="")) -> None:
    expected = os.getenv("ADMIN_PASSWORD")
    if not expected or x_admin_password != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _get_case(session: Session, session_id: str) -> Case:
    case = session.query(Case).filter(Case.session_id == session_id).first()
    if case is None:
        case = Case(session_id=session_id)
        session.add(case)
        session.commit()
        session.refresh(case)
    return case


def _log_message(session: Session, case_id: int, role: str, text: str) -> None:
    message = Message(case_id=case_id, role=role, text=text)
    session.add(message)
    session.commit()


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True})


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> JSONResponse:
    session = get_session()
    orchestrator = _build_orchestrator()
    session_id = payload.session_id or os.urandom(8).hex()
    case = _get_case(session, session_id)
    _log_message(session, case.id, "user", payload.message)
    reply, status, next_question = orchestrator.handle_turn(case, payload.message)
    case.updated_at = datetime.utcnow()
    session.add(case)
    session.commit()
    _log_message(session, case.id, "assistant", reply)
    return JSONResponse(
        {
            "session_id": session_id,
            "reply": reply,
            "case_id": case.id,
            "status": status,
            "next_question": next_question,
        }
    )


@app.post("/api/chat-with-image")
async def chat_with_image(
    session_id: str = Form(...),
    message: str = Form(...),
    image: UploadFile = File(...),
) -> JSONResponse:
    session = get_session()
    orchestrator = _build_orchestrator()
    case = _get_case(session, session_id)
    image_bytes = await image.read()
    _log_message(session, case.id, "user", message)
    storage_dir = os.path.join(os.getcwd(), "storage")
    os.makedirs(storage_dir, exist_ok=True)
    filename = f"{case.id}_{image.filename}"
    storage_path = os.path.join(storage_dir, filename)
    with open(storage_path, "wb") as handle:
        handle.write(image_bytes)
    attachment = Attachment(
        case_id=case.id,
        filename=image.filename,
        content_type=image.content_type,
        storage_path=storage_path,
    )
    session.add(attachment)
    session.commit()
    reply, status, next_question = orchestrator.handle_turn(case, message, image_bytes=image_bytes)
    case.updated_at = datetime.utcnow()
    session.add(case)
    session.commit()
    _log_message(session, case.id, "assistant", reply)
    return JSONResponse(
        {
            "session_id": session_id,
            "reply": reply,
            "case_id": case.id,
            "status": status,
            "next_question": next_question,
        }
    )


@app.get("/api/admin/cases")
def admin_cases(_: None = Depends(_admin_auth)) -> JSONResponse:
    session = get_session()
    cases = session.query(Case).order_by(Case.created_at.desc()).limit(50).all()
    payload = [
        {
            "id": case.id,
            "created_at": case.created_at.isoformat(),
            "updated_at": case.updated_at.isoformat(),
            "session_id": case.session_id,
            "category": case.category,
            "intent": case.intent,
            "decision": case.decision,
            "status": case.status,
            "discount_percent": case.discount_percent,
            "retention_step": case.retention_step,
        }
        for case in cases
    ]
    return JSONResponse(payload)


@app.get("/api/admin/cases/{case_id}")
def admin_case(case_id: int, _: None = Depends(_admin_auth)) -> JSONResponse:
    session = get_session()
    case = session.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Not found")
    messages = session.query(Message).filter(Message.case_id == case_id).all()
    attachments = session.query(Attachment).filter(Attachment.case_id == case_id).all()
    payload = {
        "case": {
            "id": case.id,
            "session_id": case.session_id,
            "category": case.category,
            "intent": case.intent,
            "decision": case.decision,
            "status": case.status,
            "discount_percent": case.discount_percent,
            "retention_step": case.retention_step,
            "reason": case.reason,
            "days_since_purchase": case.days_since_purchase,
            "purchase_date_iso": case.purchase_date_iso,
            "furniture_assembled": case.furniture_assembled,
            "electronics_defect_claimed": case.electronics_defect_claimed,
            "defect_evidence_present": case.defect_evidence_present,
            "customer_name": case.customer_name,
            "customer_phone": case.customer_phone,
            "pickup_address_json": case.pickup_address_json,
            "ticket_number": case.ticket_number,
        },
        "messages": [
            {"id": msg.id, "role": msg.role, "text": msg.text, "created_at": msg.created_at.isoformat()}
            for msg in messages
        ],
        "attachments": [
            {
                "id": attachment.id,
                "filename": attachment.filename,
                "content_type": attachment.content_type,
                "storage_path": attachment.storage_path,
            }
            for attachment in attachments
        ],
    }
    return JSONResponse(payload)
