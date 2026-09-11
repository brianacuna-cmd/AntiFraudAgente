"""Injectable port for pre-fetching a case's framed analysis pack.

Mirrors the existing ``metrics`` dependency-injection pattern: a concrete
implementation is built in ``__main__`` (via
``functools.partial(fetch_framed_analysis_pack, http_client)``) and threaded
through ``OutboxConsumer`` into ``handle_case_created``. Kept as a simple
``Callable`` alias (not a ``Protocol``) because it is a single-argument,
single-return seam — unlike ``MetricsSink``, which needs a multi-method
Protocol.
"""
from __future__ import annotations

from typing import Callable

#: ``case_id -> framed+trimmed analysis pack``. The returned string is
#: already trimmed (``domain.analysis_pack.trim_analysis_pack``) and wrapped
#: in the untrusted-data markers (``domain.policy.frame_untrusted_pack``).
#: May raise ``fraud_companion.adapters.http.errors.ApiError`` subtypes
#: (e.g. ``CaseNotFoundError``, ``CaseClosedError``) if the fetch fails.
AnalysisPackFetcher = Callable[[str], str]
