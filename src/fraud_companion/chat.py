"""Interactive REPL entrypoint for the scoring-rule authoring agent (Slice 3).

Wires ``Settings.from_env()`` -> ``AntiFraudHttpClient`` ->
``build_authoring_agent`` and drives a minimal multi-turn chat loop:
read a line from stdin, append it as a human message, invoke the agent
with the accumulated message history, print the reply, and repeat.

Kept intentionally thin — this is a developer/analyst tool, not a
production service; the underlying HTTP calls will 403 FORBIDDEN_ROLE
until the backend grants the agent key SUPERVISOR/ADMIN/AUDITOR (see
design #685). Nothing here logs a secret value.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.llm.authoring_agent import build_authoring_agent
from fraud_companion.config import Settings


def chat_main() -> None:
    """Build the authoring agent and run the REPL until EOF/Ctrl-C."""
    settings = Settings.from_env()
    http_client = AntiFraudHttpClient(
        settings.anti_fraud_base_url,
        settings.anti_fraud_agent_api_key,
        timeout=settings.http_timeout_seconds,
    )
    agent = build_authoring_agent(settings, http_client)

    messages: list = []
    print("Scoring Rule Authoring Assistant. Type your request (Ctrl-D to exit).")
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        messages.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": messages})
        messages = list(result["messages"])
        reply = messages[-1]
        print(getattr(reply, "content", reply))


if __name__ == "__main__":
    chat_main()
