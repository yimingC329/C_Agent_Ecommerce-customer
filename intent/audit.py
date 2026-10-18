from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.core.config import settings
from app.core.logger import get_logger


logger = get_logger(service="intent_audit")
_audit_lock = asyncio.Lock()


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


async def append_intent_audit(
    *,
    query: str,
    predicted_label: str,
    logic: str,
    model: str,
    confidence: float | None = None,
    confidence_source: str | None = None,
) -> None:
    """记录待人工标注的线上路由日志，不记录 API Key 或完整会话。"""
    if not settings.INTENT_AUDIT_ENABLED:
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "predicted_label": predicted_label,
        "true_label": None,
        "logic": logic,
        "model": model,
        "confidence": confidence,
        "confidence_source": confidence_source,
        "review_status": "pending",
    }
    try:
        async with _audit_lock:
            await asyncio.to_thread(
                _append_record,
                Path(settings.INTENT_AUDIT_PATH),
                record,
            )
    except Exception as exc:
        # 审计日志失败不应阻断客服主链路。
        logger.warning(f"Failed to append intent audit record: {exc}")
