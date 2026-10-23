"""Read safety-helmet candidates from alarm event folders without mutating them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class AlarmCandidate:
    event_id: str
    event_dir: Path
    original_path: Path
    metadata: Dict[str, Any]
    review: Dict[str, Any]
    semantic_bucket: str
    semantic_reason: str
    helmet_color: str


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _review_payload(event_dir: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
    stored = _read_json(event_dir / "vlm_review" / "review.json")
    review = stored.get("review")
    if isinstance(review, dict):
        return dict(review)
    return {
        "verdict": str(metadata.get("review_status") or ""),
        "evidence_quality": str(metadata.get("review_evidence_quality") or ""),
        "summary": str(metadata.get("review_summary") or ""),
        "reason_codes": list(metadata.get("review_reason_codes") or []),
        "needs_manual_review": str(metadata.get("review_status") or "") == "uncertain",
    }


def _helmet_color(summary: str) -> str:
    lowered = summary.lower()
    colors: Iterable[Tuple[Tuple[str, ...], str]] = (
        (("黄色", "yellow"), "yellow"),
        (("红色", "red"), "red"),
        (("蓝色", "blue"), "blue"),
        (("白色", "white"), "white"),
        (("橙色", "orange"), "orange"),
    )
    for terms, value in colors:
        if any(term in lowered for term in terms):
            return value
    return "uncertain"


def classify_semantic_bucket(review: Dict[str, Any]) -> Tuple[str, str]:
    verdict = str(review.get("verdict") or "").lower()
    evidence = str(review.get("evidence_quality") or "").lower()
    summary = str(review.get("summary") or "")
    reason_codes = {str(value) for value in review.get("reason_codes") or []}

    if "ordinary_hat_visible" in reason_codes:
        return "head", "ordinary_hat_visible"
    if "bare_head_visible" in reason_codes:
        return "head", "bare_head_visible"
    if verdict == "confirmed" and "helmet_visible" not in reason_codes:
        return "head", "confirmed_unhelmeted_alarm"
    if (
        "helmet_visible" in reason_codes
        and "helmet_outline_visible" in reason_codes
        and evidence in {"medium", "high"}
    ):
        return "helmet", "reviewed_helmet_visible"
    if "未佩戴安全帽" in summary or "bare head" in summary.lower():
        return "head", "summary_unhelmeted"
    return "difficult", "ambiguous_or_low_evidence"


def scan_safety_helmet_events(alarm_root: Path) -> List[AlarmCandidate]:
    candidates: List[AlarmCandidate] = []
    for event_dir in sorted(alarm_root.glob("event_*")):
        if not event_dir.is_dir():
            continue
        metadata = _read_json(event_dir / "metadata.json")
        if str(metadata.get("algorithm_code") or "") != "safety_helmet":
            continue
        original_path = event_dir / "original.jpg"
        if not original_path.is_file():
            continue
        review = _review_payload(event_dir, metadata)
        bucket, reason = classify_semantic_bucket(review)
        candidates.append(
            AlarmCandidate(
                event_id=event_dir.name,
                event_dir=event_dir,
                original_path=original_path,
                metadata=metadata,
                review=review,
                semantic_bucket=bucket,
                semantic_reason=reason,
                helmet_color=_helmet_color(str(review.get("summary") or "")),
            )
        )
    return candidates

