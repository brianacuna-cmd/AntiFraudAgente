"""Static guard: the application layer must stay prometheus-free.

Only the concrete adapter module (``adapters/metrics/prometheus_sink.py``,
Slice B) may import ``prometheus_client``. Scans application-layer source
files for the import so a future call site cannot silently reintroduce the
coupling.
"""
from __future__ import annotations

import pathlib

import fraud_companion.application as application_pkg

_APPLICATION_DIR = pathlib.Path(application_pkg.__file__).parent


def test_application_layer_does_not_import_prometheus_client():
    offending = []
    for path in _APPLICATION_DIR.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("import prometheus_client") or stripped.startswith(
                "from prometheus_client"
            ):
                offending.append(path.name)

    assert offending == []
