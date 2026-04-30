"""Persistent local memory — survives agent restarts.

Sits between the cognitive loop and the CivitasOS SDK memory: writes go to
both the local store and the remote CSP; reads hit local first, fallback to
remote.  This guarantees lessons_learned / plans / tick summaries persist
even when the backend is temporarily unreachable.

Storage is a single SQLite file at ``<data_dir>/memory.db``.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    ts    REAL NOT NULL DEFAULT (julianday('now'))
);
"""


class LocalMemory:
    """Thread-safe, file-backed key/value store using SQLite."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "memory.db"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        logger.info("LocalMemory opened at %s", self._db_path)

    # -- core API -----------------------------------------------------------

    def put(self, key: str, value: Any) -> None:
        """Store *value* (JSON-serialisable) under *key*."""
        blob = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value, ts) "
                "VALUES (?, ?, julianday('now'))",
                (key, blob),
            )
            self._conn.commit()

    def get(self, key: str) -> Any | None:
        """Return the value for *key*, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def delete(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            self._conn.commit()

    def keys(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT key FROM kv ORDER BY ts DESC").fetchall()
        return [r[0] for r in rows]

    def weighted_items(
        self,
        *,
        top_k: int = 10,
        half_life_days: float = 7.0,
    ) -> list[dict[str, Any]]:
        """Return local items ranked by exponential time-decay weight."""
        half_life_days = max(float(half_life_days), 0.001)
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value, (julianday('now') - ts) AS age_days "
                "FROM kv ORDER BY ts DESC"
            ).fetchall()
        ranked = []
        for key, raw, age_days in rows:
            age = max(float(age_days or 0.0), 0.0)
            ranked.append({
                "key": key,
                "value": json.loads(raw),
                "age_days": age,
                "decay_weight": 0.5 ** (age / half_life_days),
            })
        ranked.sort(key=lambda item: item["decay_weight"], reverse=True)
        return ranked[:top_k]

    def close(self) -> None:
        self._conn.close()


class HybridMemory:
    """Writes to both local + remote; reads local-first, remote-fallback.

    ``agent`` is the CivitasOS SDK agent (has ``remember``/``recall``).
    If *agent* is ``None`` the store operates local-only.
    """

    def __init__(self, agent: Any = None, data_dir: str | Path = "data") -> None:
        self._local = LocalMemory(data_dir)
        self._agent = agent

    # -- public API ---------------------------------------------------------

    def remember(self, key: str, value: Any) -> None:
        """Store value locally and push to CivitasOS CSP."""
        self._local.put(key, value)
        if self._agent is not None:
            try:
                self._agent.remember(key, value)
            except Exception:
                logger.debug("Remote remember(%s) failed — local-only", key)

    def recall(self, key: str) -> Any | None:
        """Read from local first; fall back to remote if missing."""
        val = self._local.get(key)
        if val is not None:
            return val
        if self._agent is not None:
            try:
                val = self._agent.recall(key)
                if val is not None:
                    # Cache remotely-fetched value locally
                    self._local.put(key, val)
                return val
            except Exception:
                logger.debug("Remote recall(%s) failed", key)
        return None

    def recall_similar(self, query: str, top_k: int = 3) -> list[Any]:
        """Semantic recall — delegates to remote CSP (no local equivalent)."""
        if self._agent is not None:
            try:
                return self._agent.recall_similar(query, top_k=top_k)
            except Exception:
                pass
        return []

    def recall_weighted(
        self,
        *,
        top_k: int = 10,
        half_life_days: float = 7.0,
    ) -> list[dict[str, Any]]:
        """Return local memories ranked by lifecycle-aware time decay."""
        return self._local.weighted_items(top_k=top_k, half_life_days=half_life_days)

    def close(self) -> None:
        self._local.close()
