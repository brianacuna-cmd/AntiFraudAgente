# Review Backlog — fraud-case-companion

Outcome of the RDD (`gentle-ai review`) reliability review of the full change
on `feat/fcc-s10-verify-fixes` (base: `develop`, ~3039 changed lines). The
reviewer re-reads the entire candidate each pass and surfaces the single worst
remaining reliability issue, so it does not converge to an approved receipt on
a candidate this size. The loop was intentionally stopped in favor of a
deliberate design pass (this document).

## Fixed and committed (test-covered, 180 tests green)

1. **Poison-message infinite redelivery** — a valid-JSON `case.created`
   envelope with a missing/empty `caseId` (or a non-dict payload) raised
   `ValueError` from `_extract_case_id` outside the consumer's guarded block,
   so the offset was never committed and Kafka redelivered forever. Now
   returns the committable `HandleResult.SKIPPED_MALFORMED`.
   (`application/case_created_handler.py`)

2. **False success without write verification** — `handle_case_created`
   returned `PROCESSED` (committing the offset) whenever `agent.invoke` did
   not raise, without confirming `put_agent_brief` actually ran. Now inspects
   the agent result for a successful `put_agent_brief` ToolMessage and raises
   the retryable `BriefNotWrittenError` otherwise.
   (`application/case_created_handler.py`)

3. **Non-dict envelope crash + error-status ToolMessage** — `event.get(...)`
   ran without checking the envelope is a dict (a JSON array/string/number
   crashed the loop); and write-verification counted any `put_agent_brief`
   ToolMessage as success without checking its `status`. Both guarded.
   (`application/case_created_handler.py`)

## OPEN — requires a design pass (do NOT patch reflexively)

### B1 — Consumer offset handling — ✅ DONE

Implemented (194 tests green). The poll loop now seeks back to the failed
offset and retries with bounded backoff (per `(topic, partition, offset)`
attempt tracking); on exhausting `kafka_max_delivery_attempts` it takes
`kafka_on_exhausted` = `crash` (default) or `skip`. `dead_letter` remains
future work (needs a producer + DLQ topic). Original finding kept below.

#### (original) Consumer offset handling breaks at-least-once (REGRESSION from fix #4)
`consumer.py` `run()` currently logs-and-`continue`s past a per-message
exception. Because `confluent_kafka.commit()` advances the per-partition
offset, a later successful message commits an offset **above** the failed one,
so the failed message is silently skipped and never redelivered — trading the
original crash-loop for silent message loss.
- **Design options:** on per-message failure, `pause()` + `seek()` back to the
  failed offset and retry with backoff; or stop the partition; or route to a
  dead-letter topic after N attempts. Decide the retry/backoff/DLQ policy
  explicitly. Fix #4 (`fix(kafka): isolate per-message failures...`) must be
  reworked, not kept as-is.

### B2 — Terminal-error handling — ✅ DONE

Implemented (183 tests green). Verified end-to-end that `create_agent`
propagates `CaseNotFoundError`/`CaseClosedError` with the default handler
(the `except` path handles production). Added a defensive result-inspection
path (`CASE_NOT_FOUND`/`CASE_CLOSED` error-status ToolMessages ->
`SKIPPED_TERMINAL`) for the `handle_tool_errors=True` mode, plus an
integration test through the real `create_agent`/ToolNode path. Original
finding text kept below for the record.

#### (original) Terminal-error assumption unproven under LangGraph ToolNode
`handle_case_created`'s `except (CaseNotFoundError, CaseClosedError)` assumes
those exceptions propagate out of `agent.invoke()`. LangGraph's `ToolNode`
(under `create_agent`) by default catches tool-body exceptions and converts
them to error-status ToolMessages, so they never reach the `except`. A case
that is genuinely not-found/closed would then finish without a written brief →
`BriefNotWrittenError` → infinite redelivery.
- **Design options:** configure `ToolNode` to re-raise for the terminal error
  classes; or inspect the agent result for terminal error-status ToolMessages
  and map them to `SKIPPED_TERMINAL` (committable). Requires confirming the
  actual `create_agent`/`ToolNode` error-handling behavior first.

## Design pass scope

Treat B1 + B2 together as one design task on the consumer/handler error
contract: define the taxonomy (retryable vs terminal vs malformed), how each
is detected given ToolNode swallows exceptions, and the offset action for each
(commit / redeliver-with-backoff / DLQ). Add tests that drive the real
`create_agent`/`ToolNode` path (not just a fake agent) for the terminal case.

## Notes
- RDD reviews left several abandoned lineages (each commit invalidated the
  prior frozen candidate). Delivery is human-owned; no receipt was burned for
  the full change.
