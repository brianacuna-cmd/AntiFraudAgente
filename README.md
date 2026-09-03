# Fraud Case Companion

An agentic service that wakes up on every `case.created` event, drafts a
first-pass analyst brief for the new fraud case using a Gemini agent, and
saves that brief back to the anti-fraud API. It **never** changes a case's
outcome: the case always stays `OPEN`, and the agent has no tool to close,
resolve, enforce, or file a SAR against it.

## What it does

1. Consumes `case.created` events from a Kafka outbox topic
   (`OutboxConsumer`, manual-commit, at-least-once).
2. For each `case.created` event, invokes a Gemini agent
   (`langchain.agents.create_agent`) with the case id and an instruction to
   analyze the case and draft its brief.
3. The agent calls the anti-fraud API through exactly 4 allow-listed tools
   and nothing else.
4. The offset is committed only after the event is fully handled
   (successfully, or determined not-for-us / terminal / malformed) — a
   transient failure is never committed, so Kafka redelivers it.

## The 4 allowed tools

| Tool | Purpose |
|---|---|
| `get_analysis_pack` | Read the case's existing analysis pack. |
| `put_agent_brief` | Write the agent's drafted brief for the case (idempotent, last-write-wins). |
| `list_cases` | List cases for context. |
| `list_aml_alerts` | List AML alerts for context. |

## Hard security locks

These are enforced in code, not just documented:

- **No re-score, resolve, enforce, or SAR tools exist.** The tool
  allow-list (`fraud_companion.domain.tools_spec.ALLOWED_TOOLS`) contains
  exactly the 4 tools above — nothing else is even defined, let alone
  callable by the agent.
- **Two independent guardrail layers** reject any disallowed tool name
  before it executes:
  - Layer 1 — a `wrap_tool_call` middleware wired into the LangGraph agent
    (`adapters/llm/guardrail.py`).
  - Layer 2 — an application-level dispatcher check
    (`application/tool_dispatcher.py`) that any tool-call path must pass
    through, independent of the agent framework.
  - Both consume the *same* `ALLOWED_TOOLS` frozenset — there is a single
    source of truth for what's allowed.
- **The case always stays `OPEN`.** `handle_case_created` never attempts a
  status change; there is no tool capable of one.
- **Authentication uses `X-Agent-Api-Key`, never `Authorization: Bearer`.**
  `AntiFraudHttpClient` sends the API key exclusively via the
  `X-Agent-Api-Key` header on every request.
- **The wake event is `case.created` only.** Any other Kafka message
  (wrong `event_type` header, or a `case.created`-headered message whose
  JSON body has a different `eventType`) is skipped, not acted on.
- **No secret is ever logged.** `Settings.__repr__`/`__str__` redact
  `google_api_key` and `anti_fraud_agent_api_key`; the entrypoint only
  logs the case id and outcome, never the settings object, the raw event
  payload, or the `X-Agent-Api-Key` header.

## Required environment variables

| Variable | Required | Default |
|---|---|---|
| `ANTI_FRAUD_BASE_URL` | yes | — |
| `ANTI_FRAUD_AGENT_API_KEY` | yes | — |
| `GOOGLE_API_KEY` | yes | — |
| `KAFKA_BOOTSTRAP_SERVERS` | yes | — |
| `KAFKA_GROUP_ID` | yes | — |
| `KAFKA_ORGANIZATION_ID` | yes | — |
| `KAFKA_OUTBOX_TOPIC` | no | `outbox.events` |
| `GEMINI_MODEL` | no | `gemini-2.5-flash` |

Missing any required variable fails fast at startup with a clear
`MissingSettingError`, never a downstream stack trace with an unset value.

## Install

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .[dev]
```

## Run tests

```bash
./.venv/bin/pytest
```

## Run

```bash
./.venv/bin/python -m fraud_companion
```

(or, once installed, the `fraud-companion` console script does the same
thing).

The process exits cleanly on `SIGINT`/`SIGTERM`: the poll loop stops and
the Kafka consumer is closed before the process terminates.
