from __future__ import annotations

import argparse

from shop_agent.gemini_client import GeminiClient
from shop_agent.orchestrator import Orchestrator
from shop_agent.policy import PolicyEngine
from shop_agent.state import load_session, save_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Shop agent CLI")
    parser.add_argument("session_id", help="Session identifier")
    parser.add_argument("message", help="User message")
    parser.add_argument("--image", help="Path to image file", default=None)
    args = parser.parse_args()

    gemini = GeminiClient()
    policy_engine = PolicyEngine.from_file(path=__policy_path())
    orchestrator = Orchestrator(gemini, policy_engine)

    state = load_session(args.session_id)
    image_bytes = None
    if args.image:
        image_bytes = open(args.image, "rb").read()
    response = orchestrator.handle_turn(state, args.message, image_bytes=image_bytes)
    save_session(state)
    print(response)


def __policy_path():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "policies.json"


if __name__ == "__main__":
    main()
