from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    return "sqlite:///./shop_agent.db"


engine = create_engine(_database_url(), future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    category: Mapped[Optional[str]] = mapped_column(String(64))
    intent: Mapped[Optional[str]] = mapped_column(String(64))
    decision: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[Optional[str]] = mapped_column(String(64))
    discount_percent: Mapped[Optional[float]] = mapped_column(Float)
    retention_step: Mapped[Optional[int]] = mapped_column(Integer)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    days_since_purchase: Mapped[Optional[int]] = mapped_column(Integer)
    purchase_date_iso: Mapped[Optional[str]] = mapped_column(String(32))
    furniture_assembled: Mapped[Optional[bool]] = mapped_column(Boolean)
    electronics_defect_claimed: Mapped[Optional[bool]] = mapped_column(Boolean)
    defect_evidence_present: Mapped[Optional[bool]] = mapped_column(Boolean)
    customer_name: Mapped[Optional[str]] = mapped_column(String(128))
    customer_phone: Mapped[Optional[str]] = mapped_column(String(64))
    pickup_address_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    ticket_number: Mapped[Optional[str]] = mapped_column(String(16))
    asked_slots: Mapped[Optional[str]] = mapped_column(Text)
    last_question_slot: Mapped[Optional[str]] = mapped_column(String(64))
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    requested_action: Mapped[Optional[str]] = mapped_column(String(64))
    user_sentiment: Mapped[Optional[str]] = mapped_column(String(32))
    emergency_trigger: Mapped[Optional[bool]] = mapped_column(Boolean)
    amazon_url: Mapped[Optional[str]] = mapped_column(String(512))
    amazon_asin: Mapped[Optional[str]] = mapped_column(String(32))
    purchase_price: Mapped[Optional[float]] = mapped_column(Float)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[int] = mapped_column(Integer, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    role: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)


class Attachment(Base):
    __tablename__ = "attachments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[int] = mapped_column(Integer, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    filename: Mapped[str] = mapped_column(String(256))
    content_type: Mapped[Optional[str]] = mapped_column(String(128))
    storage_path: Mapped[str] = mapped_column(String(512))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()
