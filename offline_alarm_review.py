"""Offline VLM review for historical alarm event folders.

The tool is intentionally isolated from the live alarm database and reporting
pipeline.  It reads alarm_outputs, calls a local OpenAI-compatible multimodal
endpoint, and writes copies plus review results to a new review_outputs folder.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import cv2
import numpy as np

import config
from helmet_review_policy import head_measurement, single_target_size, head_size_upper_bound, frame_manifest, small_target_supported
from helmet_target_evidence import build_target_evidence, verify_person


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ALARM_DIR = PROJECT_ROOT / "alarm_outputs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "review_outputs"
DEFAULT_SERVER_URL = "http://127.0.0.1:8010/v1/chat/completions"
DEFAULT_PROMPT_DIR = PROJECT_ROOT / "review" / "prompts"
DEFAULT_MODEL_PATH = Path(config.ALARM_REVIEW_MODEL_PATH)
DEFAULT_MMPROJ_PATH = Path(config.ALARM_REVIEW_MMPROJ_PATH)
PROMPT_VERSION = "v13-cellphone-action-review"

ALGORITHM_QUOTAS_100 = {
    "machinery_hazard": 70,
    "fire_detection": 20,
    "cigarette_detection": 10,
}

ALGORITHM_QUOTAS_50 = {
    "machinery_hazard": 35,
    "fire_detection": 10,
    "cigarette_detection": 5,
}

VERDICTS = {"confirmed", "rejected", "uncertain"}

REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "confidence",
        "summary",
        "reason_codes",
        "needs_manual_review",
    ],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["confirmed", "rejected", "uncertain"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "reason_codes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "needs_manual_review": {"type": "boolean"},
    },
}


@dataclass(frozen=True)
class AlarmCandidate:
    event_id: str
    event_dir: Path
    metadata_path: Path
    metadata: Dict[str, Any]

    @property
    def algorithm_code(self) -> str:
        return str(self.metadata.get("algorithm_code") or "unknown").strip()

    @property
    def algorithm_name(self) -> str:
        return str(self.metadata.get("algorithm_name") or self.metadata.get("event_type") or "未知告警").strip()

    @property
    def channel_name(self) -> str:
        return str(self.metadata.get("channel_name") or self.metadata.get("camera") or "未知通道").strip()

    @property
    def event_time(self) -> str:
        return str(self.metadata.get("time") or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用本地视觉大模型离线复核历史告警")
    parser.add_argument("--alarm-dir", type=Path, default=DEFAULT_ALARM_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reprocess-run", type=Path, default=None)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--keyframe-count", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON根节点不是对象：{path}")
    return value


def scan_candidates(alarm_dir: Path) -> List[AlarmCandidate]:
    candidates: List[AlarmCandidate] = []
    for metadata_path in sorted(alarm_dir.glob("event_*/metadata.json")):
        try:
            metadata = read_json(metadata_path)
        except Exception as exc:
            print(f"[SKIP] metadata读取失败 path={metadata_path} error={exc}", flush=True)
            continue
        event_dir = metadata_path.parent
        required_files = [
            event_dir / "original.jpg",
            event_dir / "risk_crop.jpg",
            event_dir / "alarm.jpg",
            event_dir / "alarm_clip.mp4",
        ]
        if any(not path.is_file() or path.stat().st_size <= 0 for path in required_files):
            print(f"[SKIP] 告警图片或视频不完整 path={event_dir}", flush=True)
            continue
        event_id = str(metadata.get("event_id") or event_dir.name).strip()
        candidates.append(AlarmCandidate(event_id, event_dir, metadata_path, metadata))
    return candidates


def proportional_allocations(counts: Mapping[str, int], target: int) -> Dict[str, int]:
    available = {key: int(value) for key, value in counts.items() if int(value) > 0}
    if target <= 0 or not available:
        return {key: 0 for key in available}
    target = min(target, sum(available.values()))
    total_available = sum(available.values())
    raw = {key: target * value / total_available for key, value in available.items()}
    allocated = {key: min(value, int(math.floor(raw[key]))) for key, value in available.items()}

    if target >= len(available):
        for key in available:
            allocated[key] = max(1, allocated[key])

    while sum(allocated.values()) < target:
        eligible = [key for key in available if allocated[key] < available[key]]
        key = max(eligible, key=lambda item: (raw[item] - allocated[item], available[item], item))
        allocated[key] += 1
    while sum(allocated.values()) > target:
        minimum = 1 if target >= len(available) else 0
        eligible = [key for key in available if allocated[key] > minimum]
        key = min(eligible, key=lambda item: (raw[item] - allocated[item], -allocated[item], item))
        allocated[key] -= 1
    return allocated


def evenly_spaced_sample(items: Sequence[AlarmCandidate], count: int) -> List[AlarmCandidate]:
    ordered = sorted(items, key=lambda item: (item.event_time, item.event_id))
    if count >= len(ordered):
        return list(ordered)
    if count <= 0:
        return []
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def choose_sample(candidates: Sequence[AlarmCandidate], target: int) -> List[AlarmCandidate]:
    by_algorithm: Dict[str, List[AlarmCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_algorithm[candidate.algorithm_code].append(candidate)

    if target == 100 and all(len(by_algorithm[code]) >= quota for code, quota in ALGORITHM_QUOTAS_100.items()):
        algorithm_allocations = dict(ALGORITHM_QUOTAS_100)
    elif target == 50 and all(len(by_algorithm[code]) >= quota for code, quota in ALGORITHM_QUOTAS_50.items()):
        algorithm_allocations = dict(ALGORITHM_QUOTAS_50)
    else:
        algorithm_allocations = proportional_allocations(
            {code: len(items) for code, items in by_algorithm.items()},
            target,
        )

    selected: List[AlarmCandidate] = []
    for algorithm_code, algorithm_quota in sorted(algorithm_allocations.items()):
        algorithm_items = by_algorithm[algorithm_code]
        by_channel: Dict[str, List[AlarmCandidate]] = defaultdict(list)
        for item in algorithm_items:
            by_channel[item.channel_name].append(item)
        channel_allocations = proportional_allocations(
            {channel: len(items) for channel, items in by_channel.items()},
            algorithm_quota,
        )
        for channel_name, channel_quota in sorted(channel_allocations.items()):
            selected.extend(evenly_spaced_sample(by_channel[channel_name], channel_quota))

    if len(selected) != min(target, len(candidates)):
        raise RuntimeError(f"抽样数量异常：expected={target} actual={len(selected)}")
    return sorted(selected, key=lambda item: (item.event_time, item.event_id))


def compact_metadata(candidate: AlarmCandidate) -> Dict[str, Any]:
    metadata = candidate.metadata
    compact: Dict[str, Any] = {
        "event_id": candidate.event_id,
        "algorithm_code": candidate.algorithm_code,
        "algorithm_name": candidate.algorithm_name,
        "channel_name": candidate.channel_name,
        "event_time": candidate.event_time,
        "alarm_content": metadata.get("alarm_content"),
    }
    for field in (
        "scene_mode",
        "review_pair_id",
        "crane_pair_crop_box",
        "crane_local_frame_count",
        "risk_count",
        "motion_reasons",
        "operation_evidence",
        "review_target_id",
        "primary_person_confidence",
        "primary_machinery_class",
        "primary_machinery_confidence",
        "detections",
    ):
        if field in metadata:
            compact[field] = metadata[field]
    risks = metadata.get("risks")
    if isinstance(risks, list):
        compact["risks"] = [
            {
                "person": risk.get("person"),
                "machinery": risk.get("machinery"),
                "overlap_ratio": risk.get("overlap_ratio"),
                "motion_reason": risk.get("motion_reason"),
                "candidate_region": risk.get("candidate_region"),
                "person_ground_point": risk.get("person_ground_point"),
            }
            for risk in risks[:3]
            if isinstance(risk, dict)
        ]
    original_bytes = (candidate.event_dir / "original.jpg").read_bytes()
    original = cv2.imdecode(np.frombuffer(original_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original is not None:
        compact["image_width"] = int(original.shape[1])
        compact["image_height"] = int(original.shape[0])
    target_short_edge = _safety_helmet_target_short_edge(metadata)
    if target_short_edge is not None:
        compact["head_target_short_edge_pixels"] = target_short_edge
    if candidate.algorithm_code == "safety_helmet":
        compact["head_size_source"] = (head_measurement(metadata['detections'][0])[1]
                                       if len(metadata.get('detections', [])) == 1 else 'unknown')
        compact['head_size_band'] = ('unknown' if target_short_edge is None else
                                     'under_16' if target_short_edge < 16 else
                                     '16_to_31' if target_short_edge < 32 else '32_or_more')
        if target_short_edge is None:
            compact['body_envelope_short_edge_pixels'] = head_size_upper_bound(metadata)
    return compact


def load_prompt_bundle(prompt_dir: Path) -> tuple[str, Dict[str, str], Dict[str, Any]]:
    prompt_dir = prompt_dir.resolve()
    system_prompt = (prompt_dir / "system_v2.txt").read_text(encoding="utf-8").strip()
    algorithm_prompts: Dict[str, str] = {}
    for suffix in ("_v2.txt", "_v3.txt"):
        for path in prompt_dir.glob(f"*{suffix}"):
            if path.name.startswith("system_"):
                continue
            algorithm_prompts[path.name[:-len(suffix)]] = path.read_text(encoding="utf-8").strip()
    schemas: Dict[str, Any] = {"__default__": read_json(prompt_dir / "review_schema_v2.json")}
    for path in prompt_dir.glob("review_schema_*_v3.json"):
        algorithm_code = path.name[len("review_schema_"):-len("_v3.json")]
        schemas[algorithm_code] = read_json(path)
    return system_prompt, algorithm_prompts, schemas


def review_scene_key(algorithm_code, metadata):
    if algorithm_code == "machinery_hazard" and (metadata or {}).get("scene_mode") == "crane_boom_person":
        return "crane_boom_person"
    return algorithm_code


def prompt_for(candidate: AlarmCandidate, algorithm_prompts: Mapping[str, str]) -> str:
    metadata_text = json.dumps(compact_metadata(candidate), ensure_ascii=False, separators=(",", ":"))
    task_prompt = algorithm_prompts.get(review_scene_key(candidate.algorithm_code, candidate.metadata)) or algorithm_prompts.get(
        "generic_detection",
        "复核{algorithm_name}。只有图片和关键帧中存在清晰、直接的告警证据时才confirmed。",
    )
    task_prompt = task_prompt.replace("{algorithm_name}", candidate.algorithm_name)
    return f"{task_prompt}\n候选metadata：{metadata_text}\n按照系统规则和JSON Schema返回最终结论。"


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _safety_helmet_target_short_edge(metadata: Mapping[str, Any] | None) -> int | None:
    return single_target_size(metadata)


def _contains_helmet_ambiguity(value: Any, observations: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(item or "").lower()
        for item in (
            value.get("summary") if isinstance(value, Mapping) else "",
            observations.get("spatial_relation"),
            observations.get("image_quality"),
        )
    )
    phrases = (
        "cap or helmet",
        "hat or helmet",
        "cap/helmet",
        "hat/helmet",
        "possibly a helmet",
        "maybe a helmet",
        "帽子或安全帽",
        "普通帽或安全帽",
        "可能是安全帽",
        "疑似安全帽",
    )
    return any(phrase in text for phrase in phrases)


def validate_review(
    value: Any,
    algorithm_code: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("模型结果不是JSON对象")
    model_verdict = str(value.get("verdict") or "").strip().lower()
    if model_verdict not in VERDICTS:
        raise ValueError(f"无效verdict：{model_verdict}")
    model_alarm_probability = float(value.get("alarm_probability"))
    verdict_confidence = float(value.get("verdict_confidence"))
    if not 0.0 <= model_alarm_probability <= 1.0:
        raise ValueError(f"alarm_probability超出范围：{model_alarm_probability}")
    if not 0.0 <= verdict_confidence <= 1.0:
        raise ValueError(f"verdict_confidence超出范围：{verdict_confidence}")
    evidence_quality = str(value.get("evidence_quality") or "").strip().lower()
    if evidence_quality not in {"high", "medium", "low"}:
        raise ValueError(f"无效evidence_quality：{evidence_quality}")
    reason_codes = value.get("reason_codes")
    if not isinstance(reason_codes, list):
        raise ValueError("reason_codes不是数组")
    observations = value.get("observations")
    if not isinstance(observations, dict):
        raise ValueError("observations不是对象")

    normalized_reason_codes = [str(item).strip() for item in reason_codes if str(item).strip()]
    target_short_edge = (
        _safety_helmet_target_short_edge(metadata)
        if algorithm_code == "safety_helmet"
        else None
    )
    adjustments: List[str] = []
    envelope_bound = head_size_upper_bound(metadata) if algorithm_code == 'safety_helmet' else None
    if target_short_edge is None and envelope_bound is not None and envelope_bound < 16:
        adjustments.append('helmet_body_envelope_under16:low')
        evidence_quality = 'low'
    if target_short_edge is not None and target_short_edge < 16 and evidence_quality != "low":
        adjustments.append(f"helmet_target_{target_short_edge}px:{evidence_quality}->low")
        evidence_quality = "low"
    elif target_short_edge is not None and target_short_edge < 64 and evidence_quality == "high":
        adjustments.append(f"helmet_target_{target_short_edge}px:high->medium")
        evidence_quality = "medium"
    elif algorithm_code == 'safety_helmet' and target_short_edge is None and evidence_quality == 'high':
        adjustments.append('helmet_unknown_size:high->medium')
        evidence_quality = 'medium'
    alarm_probability = model_alarm_probability
    if alarm_probability >= 0.75:
        verdict = "confirmed"
    elif alarm_probability <= 0.25:
        verdict = "rejected"
    else:
        verdict = "uncertain"
    if verdict != model_verdict:
        adjustments.append(f"probability_rule:{model_verdict}->{verdict}")
    if evidence_quality == "low" and verdict != "uncertain":
        adjustments.append(f"low_evidence:{verdict}->uncertain")
        verdict = "uncertain"
        alarm_probability = 0.5
    contradictory_motion = {
        "machinery_motion_confirmed",
        "motion_evidence_missing",
    }.issubset(set(normalized_reason_codes))
    if contradictory_motion:
        adjustments.append(f"contradictory_motion_evidence:{verdict}->uncertain")
        verdict = "uncertain"
        alarm_probability = 0.5
    risks = (metadata or {}).get("risks", [])
    static_only = bool(risks) and all(
        isinstance(risk, dict) and risk.get("motion_reason") == "static_operation_review"
        for risk in risks
    )
    if (algorithm_code == "machinery_hazard" and static_only and verdict == "confirmed"
            and "active_operation_visible" not in normalized_reason_codes):
        adjustments.append("static_operation_requires_path_b:confirmed->uncertain")
        verdict = "uncertain"
        alarm_probability = 0.5
    person_role = str(observations.get("person_role") or "").strip().lower()
    crane_scene = review_scene_key(algorithm_code, metadata) == "crane_boom_person"
    if algorithm_code == "machinery_hazard" and (
        person_role == "operator_in_cab" or "operator_in_cab" in normalized_reason_codes
    ):
        dangerous_codes = {"active_operation_visible", "dangerous_proximity", "in_machinery_path", "in_working_radius"}
        role_conflict = (
            model_verdict == "confirmed"
            or model_alarm_probability >= 0.75
            or bool(dangerous_codes.intersection(normalized_reason_codes))
            or (crane_scene and person_role != "operator_in_cab")
            or (crane_scene and observations.get("boom_person_relation") in {"under_boom", "near_boom"})
        )
        cab_fields_supported = (
            observations.get("cab_visible") == "yes"
            and observations.get("person_inside_cab") == "yes"
            and observations.get("cab_belongs_to_crane") == "yes"
            and observations.get("riding_two_wheeler") == "no"
        )
        cab_gate_verified = (metadata or {}).get("cab_verification", {}).get("verified") is True
        if role_conflict:
            adjustments.append(f"contradictory_person_role:{verdict}->uncertain")
            verdict = "uncertain"
            alarm_probability = 0.5
        elif crane_scene and (not cab_fields_supported or not cab_gate_verified):
            label = "cab_visual_evidence_missing" if not cab_fields_supported else "cab_independent_verification_missing"
            adjustments.append(f"{label}:{verdict}->uncertain")
            verdict = "uncertain"
            alarm_probability = 0.5
        elif evidence_quality == "high" and verdict_confidence >= 0.75:
            adjustments.append(f"business_rule:operator_in_cab:{verdict}->rejected")
            verdict = "rejected"
            alarm_probability = min(alarm_probability, 0.25)
        else:
            adjustments.append(f"operator_in_cab_not_high_confidence:{verdict}->uncertain")
            verdict = "uncertain"
            alarm_probability = 0.5
    if review_scene_key(algorithm_code, metadata) == "crane_boom_person" and verdict == "confirmed":
        index = observations.get("candidate_index")
        pairs = (metadata or {}).get("risks", [])[:3]
        pair_valid = (type(index) is int and 0 <= index < len(pairs)
                      and isinstance(pairs[index], dict)
                      and (pairs[index].get("machinery") or {}).get("class_id") in {1, 3}
                      and (pairs[index].get("person") or {}).get("class_id") == 0)
        checks = {
            "candidate_pair": pair_valid,
            "person": observations.get("person_visible") == "yes",
            "crane": observations.get("target_visible") == "yes",
            "crane_class": observations.get("crane_class_match") == "yes",
            "boom": observations.get("boom_visible") == "yes",
            "deployed": observations.get("crane_state") == "deployed",
            "external_worker": person_role == "pedestrian",
            "boom_relation": observations.get("boom_person_relation") in {"under_boom", "near_boom"},
            "same_space": observations.get("same_physical_space") == "yes",
            "no_barrier": observations.get("physical_barrier") == "no",
            "confidence": verdict_confidence >= 0.75,
        }
        if (metadata or {}).get("review_pair_id"):
            checks.update(
                pair_identity=observations.get("review_pair_id") == metadata["review_pair_id"],
                boom_ownership=observations.get("boom_belongs_to_crane") == "yes",
                person_gate=(metadata.get("person_verification") or {}).get("passed") is True,
            )
        missing = [key for key, passed in checks.items() if not passed]
        if missing:
            adjustments.append("crane_scene_missing_" + ",".join(missing) + ":confirmed->uncertain")
            verdict, alarm_probability = "uncertain", 0.5
    if algorithm_code == "cellphone_usage":
        attention = observations.get("attention_to_phone")
        active_use = observations.get("active_use")
        phone_position = observations.get("phone_position")
        behind_back_no_action = (
            model_verdict == "rejected"
            and observations.get("person_visible") == "yes"
            and phone_position == "behind_back"
            and active_use == "no"
            and observations.get("keyframe_consistency") == "consistent"
            and verdict_confidence >= 0.75
        )
        if behind_back_no_action and verdict != "rejected":
            adjustments.append(f"cellphone_behind_back_no_action:{verdict}->rejected")
            verdict, alarm_probability = "rejected", min(model_alarm_probability, 0.25)
        elif verdict == "confirmed":
            if (attention == "no" and active_use == "no"
                    and evidence_quality != "low" and verdict_confidence >= 0.75):
                adjustments.append("cellphone_no_use:confirmed->rejected")
                verdict, alarm_probability = "rejected", min(alarm_probability, 0.25)
            elif not (observations.get("person_visible") == "yes"
                      and observations.get("target_visible") == "yes"
                      and observations.get("same_person") == "yes"
                      and phone_position == "in_front"
                      and attention == "yes"
                      and active_use == "yes"):
                adjustments.append("cellphone_action_evidence_missing:confirmed->uncertain")
                verdict, alarm_probability = "uncertain", 0.5
    if algorithm_code == "safety_helmet":
        headgear_type = str(observations.get("headgear_type") or "").strip().lower()
        hard_shell = str(observations.get("hard_shell_visible") or "").strip().lower()
        helmet_outline = str(observations.get("helmet_outline_visible") or "").strip().lower()
        ambiguous = _contains_helmet_ambiguity(value, observations)
        if ambiguous:
            adjustments.append(f"helmet_ambiguous_description:{verdict}->uncertain")
            verdict = "uncertain"
            alarm_probability = 0.5
        elif headgear_type in {"ordinary_hat", "bare_head"}:
            if evidence_quality == "low":
                adjustments.append(f"helmet_low_evidence_{headgear_type}:{verdict}->uncertain")
                verdict = "uncertain"
                alarm_probability = 0.5
            else:
                adjustments.append(f"helmet_business_rule:{headgear_type}:{verdict}->confirmed")
                verdict = "confirmed"
                alarm_probability = max(alarm_probability, 0.75)
        elif headgear_type == "safety_helmet":
            if hard_shell == "yes" and helmet_outline == "yes":
                adjustments.append(f"helmet_business_rule:safety_helmet:{verdict}->rejected")
                verdict = "rejected"
                alarm_probability = min(alarm_probability, 0.25)
            else:
                adjustments.append(f"helmet_structure_not_confirmed:{verdict}->uncertain")
                verdict = "uncertain"
                alarm_probability = 0.5
        elif headgear_type == "unclear":
            adjustments.append(f"helmet_business_rule:unclear:{verdict}->uncertain")
            verdict = "uncertain"
            alarm_probability = 0.5
        invalid_target = (metadata or {}).get('review_target_id') is not None and value.get('target_id') != metadata['review_target_id']
        multiple_targets = len((metadata or {}).get('detections', [])) > 1
        small_or_unknown = target_short_edge is None or target_short_edge < 32
        unsupported_small = small_or_unknown and not small_target_supported(value, metadata)
        if (invalid_target or multiple_targets or unsupported_small
                or evidence_quality == 'low'
                or (verdict == 'confirmed' and (observations.get('person_visible') != 'yes' or observations.get('head_visible') != 'yes'))):
            adjustments.append('helmet_per_target_evidence_required:' + verdict + '->uncertain')
            verdict, alarm_probability = 'uncertain', .5
    needs_manual_review = bool(value.get("needs_manual_review"))
    if verdict == "uncertain" or evidence_quality == "low" or verdict_confidence < 0.75:
        needs_manual_review = True
    model_summary = str(value.get("summary") or "").strip()
    summary = model_summary
    if any(item.startswith("crane_scene_missing_") for item in adjustments):
        summary = "需人工复核：缺少同一候选人员与起重机的完整吊臂区域危险证据。原模型说明：" + model_summary
    elif any(item.startswith("contradictory_person_role:") for item in adjustments):
        summary = "需人工复核：模型的驾驶室人员标记与人员角色或危险关系矛盾。原模型说明：" + model_summary
    elif any(item.startswith(("cab_visual_evidence_missing:", "cab_independent_verification_missing:")) for item in adjustments):
        summary = "需人工复核：驾驶室内人员缺少同一目标的直接视觉证据或独立核验。原模型说明：" + model_summary
    elif any(item.startswith("business_rule:operator_in_cab:") for item in adjustments):
        summary = "规则判为拒绝：模型将人员识别为驾驶室内操作员，满足排除条件。原模型说明：" + model_summary
    elif "cellphone_no_use:confirmed->rejected" in adjustments:
        summary = "规则判为拒绝：目标人员未注视或操作手机。原模型说明：" + model_summary
    elif any(item.startswith("cellphone_behind_back_no_action:") for item in adjustments):
        summary = "规则判为拒绝：手机在目标人员身后且连续关键帧没有使用动作。原模型说明：" + model_summary
    elif "cellphone_action_evidence_missing:confirmed->uncertain" in adjustments:
        summary = "需人工复核：缺少同一人员注视并使用手机的直接图像证据。原模型说明：" + model_summary
    return {
        "model_verdict": model_verdict,
        "model_alarm_probability": round(model_alarm_probability, 4),
        "verdict": verdict,
        "alarm_probability": round(alarm_probability, 4),
        "verdict_confidence": round(verdict_confidence, 4),
        "evidence_quality": evidence_quality,
        "observations": observations,
        "summary": summary,
        "model_summary": model_summary,
        "reason_codes": normalized_reason_codes,
        "needs_manual_review": needs_manual_review,
        "policy_adjustments": adjustments,
        "target_short_edge_pixels": target_short_edge,
    }


def extract_keyframes(video_path: Path, destination: Path, count: int) -> List[Dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开告警视频：{video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            raise RuntimeError(f"告警视频帧数无效：{video_path}")
        wanted = min(max(1, int(count)), frame_count)
        positions = sorted(
            {
                min(frame_count - 1, max(0, round((index + 1) * (frame_count - 1) / (wanted + 1))))
                for index in range(wanted)
            }
        )
        keyframes: List[Dict[str, Any]] = []
        for order, frame_index in enumerate(positions, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            encoded_ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not encoded_ok:
                continue
            filename = f"frame_{order:02d}.jpg"
            output_path = destination / filename
            output_path.write_bytes(encoded.tobytes())
            keyframes.append(
                {
                    "path": output_path,
                    "filename": filename,
                    "frame_index": frame_index,
                    "seconds": round(frame_index / fps, 3) if fps > 0 else None,
                }
            )
        if not keyframes:
            raise RuntimeError(f"未能从告警视频提取关键帧：{video_path}")
        return keyframes
    finally:
        capture.release()


def create_keyframe_contact_sheet(
    keyframes: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> Path:
    tiles: List[np.ndarray] = []
    tile_width = 480
    tile_height = 270
    for index, keyframe in enumerate(keyframes, start=1):
        image_bytes = Path(keyframe["path"]).read_bytes()
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"无法读取关键帧：{keyframe['path']}")
        tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        seconds = keyframe.get("seconds")
        label = f"T{index}  {float(seconds):.2f}s" if seconds is not None else f"T{index}"
        cv2.rectangle(tile, (0, 0), (190, 34), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        raise RuntimeError("没有可用于生成拼图的关键帧")
    filler = np.zeros_like(tiles[0])
    while len(tiles) < 6:
        tiles.append(filler.copy())
    rows = [cv2.hconcat(tiles[index:index + 3]) for index in range(0, 6, 3)]
    contact_sheet = cv2.vconcat(rows)
    encoded_ok, encoded = cv2.imencode(".jpg", contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not encoded_ok:
        raise RuntimeError("关键帧拼图编码失败")
    output_path.write_bytes(encoded.tobytes())
    return output_path


def create_review_target_crop(candidate: AlarmCandidate, output_path: Path) -> Path:
    """Create a larger, unannotated context crop without replacing risk_crop.jpg."""
    image_bytes = (candidate.event_dir / "original.jpg").read_bytes()
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("无法读取复核原图")
    detections = candidate.metadata.get("detections")
    boxes: List[List[int]] = []
    if isinstance(detections, list):
        for detection in detections:
            if not isinstance(detection, Mapping):
                continue
            box = detection.get("box")
            if not isinstance(box, list) or len(box) < 4:
                continue
            try:
                boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
            except (TypeError, ValueError):
                continue
    if not boxes:
        source = candidate.event_dir / "risk_crop.jpg"
        crop = cv2.imdecode(np.frombuffer(source.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if crop is None:
            raise RuntimeError("无法读取风险区域图")
    else:
        height, width = image.shape[:2]
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        target_width = max(1, x2 - x1)
        target_height = max(1, y2 - y1)
        crop_width = max(256, target_width * 6)
        crop_height = max(256, target_height * 6)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        left = max(0, center_x - crop_width // 2)
        top = max(0, center_y - crop_height // 2)
        right = min(width, left + crop_width)
        bottom = min(height, top + crop_height)
        left = max(0, right - crop_width)
        top = max(0, bottom - crop_height)
        crop = image[top:bottom, left:right].copy()
    crop_height, crop_width = crop.shape[:2]
    shortest = min(crop_width, crop_height)
    if shortest < 512:
        scale = min(4.0, 512.0 / max(1, shortest))
        crop = cv2.resize(
            crop,
            (max(1, round(crop_width * scale)), max(1, round(crop_height * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not encoded_ok:
        raise RuntimeError("复核目标局部图编码失败")
    output_path.write_bytes(encoded.tobytes())
    return output_path


def _call_review_model_once(
    candidate: AlarmCandidate,
    keyframes: Sequence[Mapping[str, Any]],
    contact_sheet_path: Path,
    system_prompt: str,
    algorithm_prompts: Mapping[str, str],
    review_schema: Mapping[str, Any],
    server_url: str,
    timeout: float,
    max_tokens: int,
    target_crop_path: Path | None = None,
    model_name: str | None = None,
) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
    selected_schema: Mapping[str, Any]
    if "type" in review_schema:
        selected_schema = review_schema
    else:
        candidate_schema = review_schema.get(review_scene_key(candidate.algorithm_code, candidate.metadata))
        default_schema = review_schema.get("__default__")
        selected_schema = (
            candidate_schema
            if isinstance(candidate_schema, Mapping)
            else default_schema
            if isinstance(default_schema, Mapping)
            else {}
        )
    if not selected_schema:
        raise ValueError(f"没有适用于 {candidate.algorithm_code} 的复核JSON Schema")
    visual_content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt_for(candidate, algorithm_prompts)},
        {"type": "text", "text": "图片1：红框标出本次唯一目标的上下文图。" if candidate.metadata.get('helmet_context_image') else "图片1：原图中标出本次唯一人员（红框）与起重机（蓝框）。框只用于定位，不代表检测正确。" if candidate.metadata.get("crane_context_image") else "图片1：未标注完整原图。"},
        {
            "type": "image_url",
            "image_url": {"url": image_data_url(Path(candidate.metadata.get('helmet_context_image') or candidate.metadata.get('crane_context_image') or candidate.event_dir / "original.jpg"))},
        },
        {"type": "text", "text": "图片2：目标框内部的紧裁剪图；放大不会增加原始细节。" if candidate.metadata.get('helmet_context_image') else "图片2：从原图生成的复核目标局部图，包含目标周边上下文；放大不会增加原始细节。"},
        {
            "type": "image_url",
            "image_url": {
                "url": image_data_url(target_crop_path or candidate.event_dir / "risk_crop.jpg")
            },
        },
    ]
    frame_times = ", ".join(
        f"T{index}={float(item['seconds']):.2f}秒" if item.get("seconds") is not None else f"T{index}"
        for index, item in enumerate(keyframes, start=1)
    )
    temporal_label = f"图片3：按从左到右、从上到下排列的关键帧拼图。{frame_times}。"
    if candidate.metadata.get('review_pair_id'):
        temporal_label = (f"补充时序图：通过局部光流一致性筛选的同一人员裁剪，{frame_times}。仅辅助核验人体，不证明吊臂归属或危险关系。"
                          if keyframes else "补充图：原始人员裁剪的重复展示，没有可靠的跟踪关键帧。不得声称有多帧证据；清晰的静态区域危险仍可单帧判断。")
    visual_content.extend(
        [
            {"type": "text", "text": temporal_label},
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(contact_sheet_path)},
            },
        ]
    )
    if candidate.metadata.get('crane_person_crop'):
        visual_content.extend([
            {"type": "text", "text": "补充图片：仅本次人员框内部的紧裁剪，用于核验人体；不能借用其他人的特征。"},
            {"type": "image_url", "image_url": {"url": image_data_url(Path(candidate.metadata['crane_person_crop']))}},
        ])
    request_body = {
        "model": model_name or config.ALARM_REVIEW_MODEL_NAME,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": min(max_tokens, 512) if candidate.algorithm_code == "cellphone_usage" else max_tokens,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": visual_content,
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "alarm_review",
                "strict": True,
                "schema": selected_schema,
            },
        },
    }
    encoded = json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        server_url,
        data=encoded,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    raw_content = str(payload["choices"][0]["message"]["content"])
    review = validate_review(json.loads(raw_content), candidate.algorithm_code, candidate.metadata)
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return review, raw_content, dict(usage)


def call_review_model(candidate, keyframes, contact_sheet_path, system_prompt, algorithm_prompts,
                      review_schema, server_url, timeout, max_tokens, target_crop_path=None, model_name=None):
    if review_scene_key(candidate.algorithm_code, candidate.metadata) == 'crane_boom_person':
        from crane_target_review import review_pairs
        return review_pairs(candidate, keyframes, contact_sheet_path, system_prompt, algorithm_prompts,
                            review_schema, server_url, timeout, max_tokens, target_crop_path,
                            model_name or config.ALARM_REVIEW_MODEL_NAME, _call_review_model_once, verify_person)
    if candidate.algorithm_code != 'safety_helmet':
        return _call_review_model_once(candidate, keyframes, contact_sheet_path, system_prompt,
                                      algorithm_prompts, review_schema, server_url, timeout, max_tokens,
                                      target_crop_path, model_name)
    targets = [d for d in candidate.metadata.get('detections', [])
               if isinstance(d, dict) and d.get('class_id') in (0, 1)]
    manifest = frame_manifest(keyframes)
    deadline = time.monotonic()+timeout
    results, raw_results, total_usage = [], [], {}
    crop_root = Path(target_crop_path or contact_sheet_path).parent
    for index, target in enumerate(targets):
        target_id = 'person_' + str(index+1)
        meta = dict(candidate.metadata, detections=[target], review_target_id=target_id,
                    helmet_available_frames=manifest)
        child = replace(candidate, metadata=meta)
        remaining = deadline-time.monotonic()
        try:
            if remaining <= 1:
                raise TimeoutError('逐人复核总时间预算已用尽')
            crop, context, target_frames, tracking = build_target_evidence(
                candidate.event_dir/'original.jpg', target['box'], keyframes, crop_root, target_id)
            gate, gate_raw, gate_usage = verify_person(crop, target_id, server_url,
                model_name or config.ALARM_REVIEW_MODEL_NAME, max(1, deadline-time.monotonic()))
            raw_results.append(dict(target_id=target_id, stage='person_gate', response=gate_raw))
            for key, count in gate_usage.items():
                if isinstance(count, (float, int)):
                    total_usage[key] = total_usage.get(key, 0)+count
            if not gate['passed']:
                rejected = gate['rejected'] and target.get('class_id') == 0
                reviewed = dict(verdict='rejected' if rejected else 'uncertain',
                    alarm_probability=0 if rejected else .5, verdict_confidence=gate.get('confidence',0),
                    evidence_quality='medium', observations={}, reason_codes=[
                        'false_person_detection' if rejected else 'insufficient_person_evidence'],
                    summary='独立人体核验：'+str(gate.get('summary','')), needs_manual_review=not rejected,
                    policy_adjustments=['independent_person_gate'], target_short_edge_pixels=single_target_size(meta))
                raw, usage = '{}', {}
            else:
                if len(target_frames) < 2:
                    raise ValueError('同一目标的可靠跟踪关键帧不足2帧')
                from helmet_target_evidence import target_contact_sheet
                sheet = target_contact_sheet(target_frames, crop_root/(target_id+'_contact.jpg'))
                meta.update(helmet_available_frames=frame_manifest(target_frames), helmet_context_image=str(context))
                child = replace(candidate, metadata=meta)
                remaining = deadline-time.monotonic()
                if remaining <= 1:
                    raise TimeoutError('人体核验后复核时间预算已用尽')
                reviewed, raw, usage = _call_review_model_once(
                    child, target_frames, sheet, system_prompt, algorithm_prompts, review_schema,
                    server_url, remaining, max_tokens, crop, model_name)
                # Every confirmation, including large heads, needs localized temporal evidence.
                from helmet_review_policy import small_target_supported
                parsed = json.loads(raw)
                if reviewed['verdict'] == 'confirmed' and not small_target_supported(parsed, meta):
                    reviewed.update(verdict='uncertain', alarm_probability=.5, needs_manual_review=True)
                    reviewed['policy_adjustments'].append('localized_temporal_evidence_required')
            reviewed.update(person_verification=gate, target_tracking=tracking)
            raw_results.append(dict(target_id=target_id, response=raw))
            for key, count in usage.items():
                if isinstance(count, (float, int)):
                    total_usage[key] = total_usage.get(key, 0)+count
        except (OSError, ValueError, KeyError, cv2.error) as exc:
            reviewed = dict(verdict='uncertain', alarm_probability=.5, verdict_confidence=0,
                            evidence_quality='low', observations={}, reason_codes=['insufficient_evidence'],
                            summary='目标复核未完成：'+str(exc), needs_manual_review=True,
                            policy_adjustments=['target_review_failed'], target_short_edge_pixels=single_target_size(meta))
            raw_results.append(dict(target_id=target_id, error=str(exc)))
        reviewed.update(target_id=target_id, target_box=target.get('box'),
                        head_size_source=head_measurement(target)[1])
        results.append(reviewed)
    if not results:
        results.append(dict(target_id='none', verdict='uncertain', alarm_probability=.5,
                            verdict_confidence=0, evidence_quality='low', observations={},
                            reason_codes=['insufficient_evidence'], summary='无可关联的人员候选',
                            needs_manual_review=True, policy_adjustments=['missing_targets']))
    confirmed = [r for r in results if r['verdict'] == 'confirmed']
    uncertain = [r for r in results if r['verdict'] == 'uncertain']
    # Each positive must stand on its own evidence. Another person's size,
    # confidence or visibility never upgrades a target's result.
    selected = (confirmed or uncertain or results)[0]
    aggregate = dict(selected, target_reviews=results,
                     summary='；'.join(r['target_id']+'：'+r['summary'] for r in results))
    return aggregate, json.dumps(raw_results, ensure_ascii=False), total_usage


def copy_review_inputs(candidate: AlarmCandidate, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("original.jpg", "alarm.jpg", "risk_crop.jpg"):
        shutil.copy2(candidate.event_dir / filename, destination / filename)
    shutil.copy2(candidate.metadata_path, destination / "source_metadata.json")


def excluded_event_ids(manifests: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for manifest in manifests:
        rows = json.loads(manifest.resolve().read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError(f"排除清单不是数组：{manifest}")
        for row in rows:
            if isinstance(row, dict) and str(row.get("event_id") or "").strip():
                excluded.add(str(row["event_id"]).strip())
    return excluded


def write_manifest(output_dir: Path, sample: Sequence[AlarmCandidate]) -> None:
    rows = [
        {
            "index": index,
            "event_id": item.event_id,
            "algorithm_code": item.algorithm_code,
            "algorithm_name": item.algorithm_name,
            "channel_name": item.channel_name,
            "event_time": item.event_time,
            "source_dir": str(item.event_dir.resolve()),
        }
        for index, item in enumerate(sample, start=1)
    ]
    write_json(output_dir / "sample_manifest.json", rows)
    with (output_dir / "sample_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_report(output_dir: Path, results: Sequence[Dict[str, Any]]) -> None:
    verdict_counts = Counter(
        result.get("review", {}).get("verdict", "failed") if result.get("status") == "completed" else "failed"
        for result in results
    )
    algorithm_counts = Counter(result.get("algorithm_code", "unknown") for result in results)
    summary = {
        "total": len(results),
        "completed": sum(result.get("status") == "completed" for result in results),
        "failed": sum(result.get("status") != "completed" for result in results),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "algorithm_counts": dict(sorted(algorithm_counts.items())),
        "manual_review_count": sum(
            bool(result.get("review", {}).get("needs_manual_review"))
            for result in results
            if result.get("status") == "completed"
        ),
        "average_latency_seconds": round(
            sum(float(result.get("latency_seconds", 0.0)) for result in results) / max(1, len(results)),
            3,
        ),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(output_dir / "summary.json", summary)

    csv_rows: List[Dict[str, Any]] = []
    cards: List[str] = []
    for index, result in enumerate(results, start=1):
        review = result.get("review") if isinstance(result.get("review"), dict) else {}
        verdict = str(review.get("verdict") or "failed")
        event_id = str(result.get("event_id") or "")
        event_rel = f"events/{event_id}"
        contact_sheet = str(result.get("keyframes_contact_sheet") or "")
        contact_figure = (
            f'<figure><img loading="lazy" src="{event_rel}/{html.escape(contact_sheet)}">'
            f'<figcaption>模型输入的关键帧拼图</figcaption></figure>'
            if contact_sheet
            else ""
        )
        keyframe_figures = "".join(
            f'<figure><img loading="lazy" src="{event_rel}/keyframes/{html.escape(str(item.get("filename", "")))}">'
            f'<figcaption>关键帧 {position} · {html.escape(str(item.get("seconds", "")))}s</figcaption></figure>'
            for position, item in enumerate(result.get("keyframes") or [], start=1)
        )
        csv_rows.append(
            {
                "index": index,
                "event_id": event_id,
                "algorithm_name": result.get("algorithm_name", ""),
                "channel_name": result.get("channel_name", ""),
                "event_time": result.get("event_time", ""),
                "status": result.get("status", ""),
                "model_verdict": review.get("model_verdict", ""),
                "model_alarm_probability": review.get("model_alarm_probability", ""),
                "verdict": verdict,
                "alarm_probability": review.get("alarm_probability", ""),
                "verdict_confidence": review.get("verdict_confidence", ""),
                "evidence_quality": review.get("evidence_quality", ""),
                "needs_manual_review": review.get("needs_manual_review", ""),
                "summary": review.get("summary", ""),
                "reason_codes": "|".join(review.get("reason_codes") or []),
                "policy_adjustments": "|".join(review.get("policy_adjustments") or []),
                "latency_seconds": result.get("latency_seconds", ""),
                "error": result.get("error", ""),
                "source_dir": result.get("source_dir", ""),
            }
        )
        cards.append(
            f"""
            <article class="card {html.escape(verdict)}">
              <header><span class="number">#{index}</span><strong>{html.escape(str(result.get('algorithm_name', '')))}</strong>
                <span class="verdict">{html.escape(verdict)}</span>
                <span>告警概率 {html.escape(str(review.get('alarm_probability', '')))}</span>
                <span>结论置信度 {html.escape(str(review.get('verdict_confidence', '')))}</span>
                <span>证据 {html.escape(str(review.get('evidence_quality', '')))}</span></header>
              <p>{html.escape(str(result.get('channel_name', '')))} · {html.escape(str(result.get('event_time', '')))}</p>
              <p class="summary">{html.escape(str(review.get('summary') or result.get('error') or ''))}</p>
              <p class="reasons">{html.escape(', '.join(review.get('reason_codes') or []))}</p>
              <div class="images">
                <figure><img loading="lazy" src="{event_rel}/original.jpg"><figcaption>原图</figcaption></figure>
                <figure><img loading="lazy" src="{event_rel}/alarm.jpg"><figcaption>原告警图</figcaption></figure>
                <figure><img loading="lazy" src="{event_rel}/risk_crop.jpg"><figcaption>风险局部</figcaption></figure>
                {contact_figure}
              </div>
              <div class="keyframes">{keyframe_figures}</div>
              <footer><a href="{event_rel}/review.json">review.json</a> · {html.escape(event_id)}</footer>
            </article>
            """
        )

    if csv_rows:
        with (output_dir / "review_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-VL历史告警复核结果</title><style>
body{{font-family:"Microsoft YaHei",sans-serif;background:#f3f5f8;color:#1f2937;margin:0;padding:24px}}
.top{{position:sticky;top:0;background:#fff;padding:16px 20px;border-radius:12px;box-shadow:0 4px 18px #0001;z-index:2}}
.grid{{display:grid;gap:18px;margin-top:20px}}.card{{background:#fff;border-left:6px solid #64748b;border-radius:12px;padding:16px;box-shadow:0 3px 14px #0001}}
.card.confirmed{{border-color:#16a34a}}.card.rejected{{border-color:#dc2626}}.card.uncertain{{border-color:#d97706}}.card.failed{{border-color:#6b7280}}
header{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}.number{{color:#64748b}}.verdict{{font-weight:700;text-transform:uppercase}}.summary{{font-size:16px}}.reasons{{color:#64748b}}
.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}.keyframes{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:12px}}figure{{margin:0}}img{{width:100%;height:260px;object-fit:contain;background:#111;border-radius:8px}}.keyframes img{{height:150px}}figcaption{{text-align:center;color:#64748b;margin-top:4px}}
footer{{margin-top:10px;color:#64748b;font-size:13px}}a{{color:#2563eb}}@media(max-width:900px){{.images,.keyframes{{grid-template-columns:1fr}}img,.keyframes img{{height:auto}}}}
</style></head><body><section class="top"><h1>Qwen3-VL 4B 历史告警离线复核</h1>
<p>总数 {summary['total']} · 完成 {summary['completed']} · 失败 {summary['failed']} · 人工复核 {summary['manual_review_count']} · 平均耗时 {summary['average_latency_seconds']} 秒</p>
<p>确认 {verdict_counts.get('confirmed', 0)} · 驳回 {verdict_counts.get('rejected', 0)} · 不确定 {verdict_counts.get('uncertain', 0)}</p></section>
<main class="grid">{''.join(cards)}</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def reprocess_review_run(output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    manifest_rows = json.loads((output_dir / "sample_manifest.json").read_text(encoding="utf-8-sig"))
    order = {
        str(row.get("event_id") or ""): int(row.get("index") or position)
        for position, row in enumerate(manifest_rows, start=1)
        if isinstance(row, dict)
    }
    results: List[Dict[str, Any]] = []
    for review_path in (output_dir / "events").glob("*/review.json"):
        result = read_json(review_path)
        raw_path = review_path.parent / "raw_response.txt"
        if result.get("status") == "completed" and raw_path.is_file():
            raw_value = json.loads(raw_path.read_text(encoding="utf-8"))
            result["review"] = validate_review(raw_value, str(result.get("algorithm_code") or ""))
            result["reprocessed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            write_json(review_path, result)
        results.append(result)
    results.sort(key=lambda item: order.get(str(item.get("event_id") or ""), 10**9))
    generate_report(output_dir, results)
    print(f"REPROCESS_COMPLETE output={output_dir} count={len(results)}", flush=True)
    return 0


def run() -> int:
    args = parse_args()
    if args.reprocess_run is not None:
        return reprocess_review_run(args.reprocess_run)
    if args.count < 1:
        raise ValueError("--count必须大于0")
    if args.keyframe_count < 1:
        raise ValueError("--keyframe-count必须大于0")
    system_prompt, algorithm_prompts, review_schema = load_prompt_bundle(args.prompt_dir)
    excluded = excluded_event_ids(args.exclude_manifest)
    candidates = scan_candidates(args.alarm_dir.resolve())
    candidates = [candidate for candidate in candidates if candidate.event_id not in excluded]
    if len(candidates) < args.count:
        raise RuntimeError(f"可用告警不足：required={args.count} available={len(candidates)}")
    sample = choose_sample(candidates, args.count)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or (args.output_root / f"alarm_vlm_keyframes_review_{args.count}_{timestamp}")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "events").mkdir()
    shutil.copytree(args.prompt_dir.resolve(), output_dir / "prompts_used")
    write_manifest(output_dir, sample)
    write_json(
        output_dir / "run_info.json",
        {
            "mode": "offline_shadow_review",
            "prompt_version": PROMPT_VERSION,
            "prompt_dir": str(args.prompt_dir.resolve()),
            "keyframe_count": args.keyframe_count,
            "live_alarm_database_modified": False,
            "reporting_triggered": False,
            "alarm_dir": str(args.alarm_dir.resolve()),
            "output_dir": str(output_dir),
            "server_url": args.server_url,
            "model_path": str(DEFAULT_MODEL_PATH),
            "mmproj_path": str(DEFAULT_MMPROJ_PATH),
            "sample_count": len(sample),
            "selection_method": "algorithm quota + proportional channel allocation + evenly spaced time sample",
            "excluded_event_count": len(excluded),
            "excluded_manifests": [str(path.resolve()) for path in args.exclude_manifest],
            "algorithm_quotas": dict(Counter(item.algorithm_code for item in sample)),
            "channel_quotas": dict(Counter(item.channel_name for item in sample)),
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )
    print(f"OUTPUT_DIR={output_dir}", flush=True)
    print(f"SAMPLE algorithms={dict(Counter(item.algorithm_code for item in sample))}", flush=True)
    print(f"SAMPLE channels={dict(Counter(item.channel_name for item in sample))}", flush=True)
    if args.dry_run:
        print("DRY_RUN_COMPLETE", flush=True)
        return 0

    results: List[Dict[str, Any]] = []
    batch_started = time.monotonic()
    for index, candidate in enumerate(sample, start=1):
        destination = output_dir / "events" / candidate.event_id
        copy_review_inputs(candidate, destination)
        started = time.monotonic()
        review: Dict[str, Any] | None = None
        raw_response = ""
        usage: Dict[str, Any] = {}
        error = ""
        keyframes: List[Dict[str, Any]] = []
        contact_sheet_path = destination / "keyframes_contact.jpg"
        try:
            keyframes = extract_keyframes(
                candidate.event_dir / "alarm_clip.mp4",
                destination / "keyframes",
                args.keyframe_count,
            )
            create_keyframe_contact_sheet(keyframes, contact_sheet_path)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if not error:
            for attempt in range(args.retries + 1):
                try:
                    review, raw_response, usage = call_review_model(
                        candidate,
                        keyframes=keyframes,
                        contact_sheet_path=contact_sheet_path,
                        system_prompt=system_prompt,
                        algorithm_prompts=algorithm_prompts,
                        review_schema=review_schema,
                        server_url=args.server_url,
                        timeout=args.timeout,
                        max_tokens=args.max_tokens,
                    )
                    error = ""
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.retries:
                        time.sleep(1.0)
        latency = round(time.monotonic() - started, 3)
        result: Dict[str, Any] = {
            "event_id": candidate.event_id,
            "algorithm_code": candidate.algorithm_code,
            "algorithm_name": candidate.algorithm_name,
            "channel_name": candidate.channel_name,
            "event_time": candidate.event_time,
            "source_dir": str(candidate.event_dir.resolve()),
            "status": "completed" if review is not None else "failed",
            "review": review,
            "prompt_version": PROMPT_VERSION,
            "keyframes": [
                {key: value for key, value in item.items() if key != "path"}
                for item in keyframes
            ],
            "keyframes_contact_sheet": contact_sheet_path.name if contact_sheet_path.is_file() else "",
            "latency_seconds": latency,
            "usage": usage,
            "error": error,
            "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        write_json(destination / "review.json", result)
        (destination / "raw_response.txt").write_text(raw_response or error, encoding="utf-8")
        results.append(result)
        generate_report(output_dir, results)
        verdict = review.get("verdict") if review else "failed"
        elapsed = time.monotonic() - batch_started
        average = elapsed / index
        eta = average * (len(sample) - index)
        print(
            f"[{index:03d}/{len(sample):03d}] {candidate.algorithm_code} {candidate.channel_name} "
            f"verdict={verdict} latency={latency:.1f}s eta={eta/60:.1f}m event={candidate.event_id}",
            flush=True,
        )

    generate_report(output_dir, results)
    print(f"REVIEW_COMPLETE output={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("INTERRUPTED", file=sys.stderr, flush=True)
        raise SystemExit(130)
