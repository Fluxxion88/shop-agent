from __future__ import annotations

import argparse

from shop_agent.db import Message, get_session, init_db
from shop_agent.gemini_client import GeminiClient
from shop_agent.orchestrator import DialogManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Shop agent CLI")
    parser.add_argument("session_id", help="Session identifier")
    parser.add_argument("message", help="User message")
    parser.add_argument("--image", help="Path to image file", default=None)
    args = parser.parse_args()

    init_db()
    gemini = GeminiClient()
    orchestrator = DialogManager(gemini)
    session = get_session()
    case = session.query(__case_model()).filter_by(session_id=args.session_id).first()
    if case is None:
        case = __case_model()(session_id=args.session_id)
        session.add(case)
        session.commit()
        session.refresh(case)
    session.add(Message(case_id=case.id, role="user", text=args.message))
    session.commit()
    image_bytes = None
    if args.image:
        image_bytes = open(args.image, "rb").read()
    response, _, _ = orchestrator.handle_turn(case, args.message, image_bytes=image_bytes)
    session.add(case)
    session.commit()
    session.add(Message(case_id=case.id, role="assistant", text=response))
    session.commit()
    print(response)


def __case_model():
    from shop_agent.db import Case

    return Case


if __name__ == "__main__":
    main()
