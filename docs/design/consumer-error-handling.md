# Design Pass — Consumer Offset & Error Handling (B1 + B2)

Scope: resolve the two open findings in [../review-backlog.md](../review-backlog.md)
as one coherent error contract for the `case.created` pipeline
(`adapters/kafka/consumer.py` + `application/case_created_handler.py`).

Verified against the installed stack: langgraph 1.2.11, langgraph-prebuilt
1.1.0, langchain 1.3.18, langchain-google-genai 4.4.0, confluent-kafka wheel.

## Current outcome taxonomy (keep)

`handle_case_created` returns (never raises) for outcomes the offset may be
committed on: `SKIPPED_IGNORED`, `SKIPPED_TERMINAL`, `SKIPPED_MALFORMED`,
`PROCESSED`. It raises only for **retryable** failures (transient network/5xx,
auth/config defects, `BriefNotWrittenError`). The consumer commits iff
`handle_case_created` returned without raising. This taxonomy is correct and
stays; B1/B2 only fix how the two layers act on it.

## B1 — Offset handling breaks at-least-once (must rework fix #4)

### Problem
`enable.auto.commit=false` is set and `commit(msg)` commits the per-partition
offset `msg.offset + 1`. Offsets are per-partition monotonic, so a later
successful commit supersedes any earlier uncommitted one. The current
`run()` "log-and-`continue`" therefore lets a failed message's offset be
overtaken by a later success → the failed (retryable) message is silently
skipped, never redelivered. This is worse than the crash it replaced.

### Key fact
After a failed message we must NOT let the consumer's read position advance
past it. With manual commit, `poll()` still advances the in-memory position,
so re-consuming requires an explicit `seek()` back to `msg.offset`.

### Options
1. **Pause + seek + backoff (single-worker, in-process retry).** On a
   retryable exception: `seek(TopicPartition(msg.topic(), msg.partition(),
   msg.offset()))`, sleep a bounded backoff, keep polling (re-reads the same
   message). No commit until success. Correct at-least-once. Head-of-line
   blocking is acceptable here because only genuinely *retryable* exceptions
   reach the loop (terminal/malformed already returned + committed).
2. **Retry N then dead-letter (DLQ).** Same as (1) but after N attempts,
   produce the message to a dead-letter topic and commit so the partition
   progresses. Needs a producer + DLQ topic. Best production semantics.
3. **Crash-fast (revert fix #4).** Let the exception stop the consumer. No
   data loss, operationally simple, but one bad message downs the service.

### Recommendation
Ship **Option 1 now**, evolve to **Option 2** when a producer/DLQ is
provisioned. Bound Option 1 with a per-`(partition, offset)` attempt counter
and a configurable `max_delivery_attempts`; on exhaustion take a configurable
terminal action (`crash` by default — surfaces mis-classified permanent errors
loudly; `dead_letter` once DLQ exists). Add config:
`kafka_max_delivery_attempts` (int), `kafka_retry_backoff_seconds` (float),
`kafka_on_exhausted` (`crash` | `skip` | `dead_letter`). Never silently skip.

### Sketch
```
try:
    self.process_message(msg)   # commits internally on non-raising outcomes
except Exception:
    attempts = self._bump_attempts(msg)          # keyed by (topic, partition, offset)
    if attempts >= self._max_delivery_attempts:
        self._on_exhausted(msg)                  # crash | dead-letter | explicit skip+commit
    else:
        self._consumer.seek(TopicPartition(msg.topic(), msg.partition(), msg.offset()))
        time.sleep(self._backoff(attempts))
```

## B2 — Terminal-error handling (reviewer premise CORRECTED)

### What the reviewer claimed
LangGraph `ToolNode` "by default catches exceptions raised during tool
execution and turns them into error-status ToolMessages", so the handler's
`except (CaseNotFoundError, CaseClosedError)` is dead and terminal cases loop
forever.

### What the installed code actually does (verified by source + introspection)
`ToolNode(handle_tool_errors=_default_handle_tool_errors)` is the default.
`_default_handle_tool_errors(e)` returns a message **only** for
`ToolInvocationError` (argument/validation errors) and executes `raise e` for
everything else. Both `_execute_tool_sync` and `_run_one` route a callable
`handle_tool_errors` through `flag(e)`, so a non-`ToolInvocationError` raised
in a tool body (e.g. `CaseNotFoundError`, `CaseClosedError`, `ConnectionError`)
**propagates out of `agent.invoke()`**. The guardrail middleware
`wrap_tool_call` is transparent (`return execute(request)`), so it does not
change this. => The terminal `except` is reachable and correct today. The
reviewer's B2 premise is false for this version.

### Real residual risks (these are what to harden)
- **Fragile implicit dependency.** Correctness hinges on the default handler
  re-raising. If anyone sets `handle_tool_errors=True`/str, ALL tool
  exceptions become error-status ToolMessages, the terminal `except` goes
  dead, `_brief_was_written` returns False, and genuinely-terminal cases
  raise `BriefNotWrittenError` → infinite redelivery.
- **Argument `ValidationError`** on a tool call becomes a `ToolInvocationError`
  → error-status ToolMessage (not raised). For `put_agent_brief` this surfaces
  as "brief not written" → redelivery, which is acceptable (bad args are a
  bug) but should be understood.

### Design
Make terminal handling robust to **both** propagation modes (belt + suspenders):
1. Keep `except (CaseNotFoundError, CaseClosedError) -> SKIPPED_TERMINAL`.
2. Also, after a non-raising run, inspect the agent result for a terminal
   error-status ToolMessage (content/name indicating case-not-found / closed)
   and map it to `SKIPPED_TERMINAL` **before** the `_brief_was_written` check.
   This keeps the pipeline correct regardless of the `handle_tool_errors`
   setting.
3. Document a hard rule: do not change `ToolNode`'s `handle_tool_errors`
   default without updating this logic. Consider pinning behavior with a test.

## Test plan (strict TDD)

- **B1**: consumer test — `poll` yields a failing message; assert `seek` called
  with `(topic, partition, offset)`, no commit, backoff applied; then the
  re-polled message succeeds and commits. Separate test: attempts reach
  `max_delivery_attempts` → configured terminal action taken (crash raises /
  dead_letter produces / skip commits with a CRITICAL log). No silent skip.
- **B2**: a test through the **real** `create_agent` path using a fake chat
  model that emits a `get_analysis_pack` tool call whose bound tool raises
  `CaseNotFoundError`; assert `handle_case_created` returns `SKIPPED_TERMINAL`
  (proves propagation against the actual lib and guards lib upgrades). Second
  test: an error-status terminal ToolMessage in the result → `SKIPPED_TERMINAL`
  via result inspection (covers the `handle_tool_errors=True` mode).

## Known tradeoff — head-of-line blocking (accepted)

The B1 retry is a synchronous `seek()` + `sleep()` on the single poll thread.
On a multi-partition subscription, a retryable failure on one message delays
processing of all other partitions for the cumulative backoff of that message's
retries. This is a **deliberate choice of correctness (at-least-once, no silent
skip) over cross-partition availability**, and it is **bounded**, not infinite:

- `kafka_on_exhausted=crash` (default): after `kafka_max_delivery_attempts` the
  consumer raises and the loop exits — total block ≈ `backoff * (1+2+…+(N-1))`
  before crash (≈10s with defaults), not indefinite.
- `kafka_on_exhausted=skip`: after exhaustion the message is committed and the
  loop moves on immediately.

This is acceptable for a single- or low-partition outbox consumer. If throughput
across many partitions matters, the resolution is a larger change (out of scope
here): per-partition `pause()`/`resume()` so only the failing partition backs
off, or an async/DLQ path so the poll thread never blocks. Tracked as future
work alongside the DLQ item. Flagged by RDD review (R3-blocking-retry-loop) and
accepted as a conscious tradeoff, not a defect.

## Effort / sequencing
Moderate. Land B2 first (small: result-inspection helper + two tests; no infra),
then B1 Option 1 (config + attempt tracking + seek/backoff). Defer B1 Option 2
(DLQ) until a Kafka producer is introduced. Both are isolated to the two files
plus `config.py`; no change to the 4-tool contract, auth, or event name.
