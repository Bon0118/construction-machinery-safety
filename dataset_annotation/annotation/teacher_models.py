"""Independent teacher models used by the safety-helmet v2 annotation pipeline.

Heavy dependencies are imported lazily so rule tests and the online YOLOv5
service do not need the isolated annotation environment.
"""

from __future__ import annotations

import base64
import gc
import json
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from dataset_annotation.annotation.rules import Box, box_iou


@dataclass(frozen=True)
class TeacherCandidate:
    candidate_id: str
    kind: str
    score: float
    box: Box
    source: str
    prompt: str = ""
    sam_refined: bool = False
    mask_score: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["box"] = list(self.box)
        value["score"] = round(float(self.score), 5)
        if self.mask_score is not None:
            value["mask_score"] = round(float(self.mask_score), 5)
        return value


@dataclass(frozen=True)
class SemanticDecision:
    is_target_person: bool
    head_state: str
    selected_candidate_id: str
    direct_box: Tuple[int, ...]
    confidence: float
    helmet_color: str
    wearing_state: str
    visibility_ratio: float
    truncated: bool
    reason: str
    source: str = "qwen3vl4b_candidate_selection"

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["confidence"] = round(float(self.confidence), 5)
        value["visibility_ratio"] = round(float(self.visibility_ratio), 5)
        return value


def _nms(items: Sequence[TeacherCandidate], threshold: float) -> List[TeacherCandidate]:
    pending = sorted(items, key=lambda item: item.score, reverse=True)
    kept: List[TeacherCandidate] = []
    while pending:
        current = pending.pop(0)
        kept.append(current)
        pending = [item for item in pending if box_iou(current.box, item.box) < threshold]
    return kept


def _release_torch(model: Any = None) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _resolved_device(preferred: str) -> str:
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class IndependentPersonDetector:
    """COCO person detector which does not use the safety-helmet student model."""

    def __init__(self, config: Mapping[str, Any], preferred_device: str) -> None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        self.torch = torch
        self.config = dict(config)
        self.device = _resolved_device(preferred_device)
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.transform = self.weights.transforms()
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=self.weights,
            box_score_thresh=float(self.config["confidence_threshold"]),
        ).eval().to(self.device)

    def predict(self, image: np.ndarray) -> List[TeacherCandidate]:
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).to(self.device)
        with self.torch.inference_mode():
            output = self.model([tensor])[0]
        threshold = float(self.config["confidence_threshold"])
        candidates: List[TeacherCandidate] = []
        for score, label, box in zip(output["scores"], output["labels"], output["boxes"]):
            if int(label.item()) != 1 or float(score.item()) < threshold:
                continue
            values = [int(round(float(value))) for value in box.tolist()]
            if values[2] <= values[0] or values[3] <= values[1]:
                continue
            candidates.append(
                TeacherCandidate(
                    candidate_id="",
                    kind="person",
                    score=float(score.item()),
                    box=tuple(values),
                    source="torchvision_fasterrcnn_resnet50_fpn_v2_coco",
                    prompt="person",
                )
            )
        candidates = _nms(candidates, float(self.config["nms_iou_threshold"]))
        limit = int(self.config["maximum_people_per_image"])
        return [replace(item, candidate_id=f"P{index:02d}") for index, item in enumerate(candidates[:limit], 1)]

    def close(self) -> None:
        model = self.model
        self.model = None
        _release_torch(model)


def _candidate_kind(label: str) -> str:
    normalized = label.lower().replace("_", " ").strip()
    if any(token in normalized for token in ("motorcycle helmet", "welding mask")):
        return "other_headgear"
    if any(token in normalized for token in ("construction safety helmet", "hard hat", "safety hat")):
        return "helmet_shell"
    if "ordinary hat" in normalized:
        return "other_headgear"
    if "helmet" in normalized:
        return "helmet_shell"
    if any(token in normalized for token in ("head", "face")):
        return "head"
    if "person" in normalized or "human" == normalized:
        return "person"
    return "other_headgear"


class GroundingDinoLocator:
    """Open-vocabulary head and headgear candidate locator."""

    def __init__(self, config: Mapping[str, Any], preferred_device: str) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.config = dict(config)
        self.device = _resolved_device(preferred_device)
        model_id = str(self.config["model_id"])
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Grounding DINO's text-enhancer path in transformers 4.48 can promote
        # hidden states to FP32 even when the checkpoint is loaded as FP16.
        # Keeping this comparatively small teacher model in FP32 avoids a
        # Float/Half matmul mismatch and still fits comfortably on the RTX 4060.
        dtype = torch.float32
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).eval().to(self.device)
        self.prompt_text = ". ".join(str(item) for item in self.config["prompts"]) + "."

    def predict(self, image: np.ndarray) -> List[TeacherCandidate]:
        from PIL import Image

        height, width = image.shape[:2]
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil_image, text=self.prompt_text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=float(self.config["box_threshold"]),
            text_threshold=float(self.config["text_threshold"]),
            target_sizes=[(height, width)],
        )[0]
        text_labels = result.get("text_labels")
        if text_labels is None:
            text_labels = result.get("labels")
        if text_labels is None:
            text_labels = []
        grouped: Dict[str, List[TeacherCandidate]] = {}
        for score, label, box in zip(result["scores"], text_labels, result["boxes"]):
            label_text = str(label)
            values = [int(round(float(value))) for value in box.tolist()]
            values[0] = max(0, min(width - 1, values[0]))
            values[1] = max(0, min(height - 1, values[1]))
            values[2] = max(values[0] + 1, min(width, values[2]))
            values[3] = max(values[1] + 1, min(height, values[3]))
            kind = _candidate_kind(label_text)
            grouped.setdefault(kind, []).append(
                TeacherCandidate(
                    candidate_id="",
                    kind=kind,
                    score=float(score.item()),
                    box=tuple(values),
                    source="grounding_dino_tiny",
                    prompt=label_text,
                )
            )
        result_items: List[TeacherCandidate] = []
        prefix = {"head": "H", "helmet_shell": "K", "person": "D", "other_headgear": "O"}
        maximum = int(self.config["maximum_candidates_per_kind"])
        for kind, items in grouped.items():
            kept = _nms(items, float(self.config["nms_iou_threshold"]))[:maximum]
            result_items.extend(
                replace(item, candidate_id=f"{prefix.get(kind, 'X')}{index:02d}")
                for index, item in enumerate(kept, 1)
            )
        return result_items

    def close(self) -> None:
        model = self.model
        self.model = None
        self.processor = None
        _release_torch(model)


def mask_box(mask: np.ndarray) -> Box | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class Sam2Refiner:
    """SAM2.1 box-prompt refinement for Grounding DINO candidates."""

    def __init__(self, config: Mapping[str, Any], preferred_device: str) -> None:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.config = dict(config)
        self.device = _resolved_device(preferred_device)
        checkpoint = Path(str(self.config["checkpoint"])).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint不存在：{checkpoint}")
        model = build_sam2(str(self.config["model_cfg"]), str(checkpoint), device=self.device)
        self.predictor = SAM2ImagePredictor(model)

    def refine(self, image: np.ndarray, candidates: Sequence[TeacherCandidate]) -> List[TeacherCandidate]:
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        refined: List[TeacherCandidate] = []
        for item in candidates:
            if item.kind not in {"head", "helmet_shell", "other_headgear"}:
                refined.append(item)
                continue
            box_array = np.asarray(item.box, dtype=np.float32)
            with self.torch.inference_mode():
                masks, scores, _logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_array,
                    multimask_output=False,
                )
            refined_box = mask_box(masks[0])
            if refined_box is None:
                refined.append(item)
                continue
            original_area = max(1, item.box[2] - item.box[0]) * max(1, item.box[3] - item.box[1])
            refined_area = max(1, refined_box[2] - refined_box[0]) * max(1, refined_box[3] - refined_box[1])
            if (
                box_iou(item.box, refined_box) < float(self.config["minimum_mask_box_iou"])
                or refined_area > original_area * float(self.config["maximum_mask_area_multiplier"])
            ):
                refined.append(item)
                continue
            refined.append(
                replace(
                    item,
                    box=refined_box,
                    source=f"{item.source}+sam2.1_hiera_tiny",
                    sam_refined=True,
                    mask_score=float(scores[0]),
                )
            )
        return refined

    def close(self) -> None:
        predictor = self.predictor
        self.predictor = None
        _release_torch(predictor)


def estimated_head_box(person: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = person
    person_width = max(1, x2 - x1)
    person_height = max(1, y2 - y1)
    head_width = max(20, int(round(person_width * 0.62)))
    head_height = max(20, int(round(min(person_height * 0.26, head_width * 1.28))))
    center_x = (x1 + x2) / 2.0
    return (
        max(0, int(round(center_x - head_width / 2))),
        max(0, y1),
        min(width, int(round(center_x + head_width / 2))),
        min(height, y1 + head_height),
    )


def union_box(first: Box, second: Box, width: int, height: int) -> Box:
    return (
        max(0, min(first[0], second[0])),
        max(0, min(first[1], second[1])),
        min(width, max(first[2], second[2])),
        min(height, max(first[3], second[3])),
    )


def candidate_matches_target_person(candidate: Box, person: Box) -> bool:
    """Require a head/headgear box to belong to the target person's upper body."""
    center_x = (candidate[0] + candidate[2]) / 2.0
    center_y = (candidate[1] + candidate[3]) / 2.0
    width = max(1, person[2] - person[0])
    height = max(1, person[3] - person[1])
    candidate_width = max(1, candidate[2] - candidate[0])
    horizontal_overlap = max(0, min(candidate[2], person[2]) - max(candidate[0], person[0]))
    return (
        person[0] - width * 0.03 <= center_x <= person[2] + width * 0.03
        and person[1] - height * 0.10 <= center_y <= person[1] + height * 0.45
        and horizontal_overlap / candidate_width >= 0.30
    )


def build_label_candidates(
    image: np.ndarray,
    person: Box,
    grounded: Sequence[TeacherCandidate],
) -> List[TeacherCandidate]:
    """Convert raw grounding results into mutually exclusive head/helmet label candidates."""
    height, width = image.shape[:2]
    relevant = [item for item in grounded if candidate_matches_target_person(item.box, person)]
    heads = [item for item in relevant if item.kind == "head"]
    shells = [item for item in relevant if item.kind == "helmet_shell"]
    other_headgear = [item for item in relevant if item.kind == "other_headgear"]
    estimate = TeacherCandidate(
        candidate_id="H00",
        kind="head",
        score=0.20,
        box=estimated_head_box(person, width, height),
        source="person_anatomical_geometry_fallback",
        prompt="estimated whole head",
    )
    head_candidates = heads or [estimate]
    final: List[TeacherCandidate] = []
    for index, item in enumerate(head_candidates, 1):
        final.append(replace(item, candidate_id=f"H{index:02d}", kind="head"))
    head_index = len(final)
    for headgear in other_headgear:
        head = min(
            head_candidates,
            key=lambda item: abs((item.box[0] + item.box[2] - headgear.box[0] - headgear.box[2]) / 2.0)
            + abs((item.box[1] + item.box[3] - headgear.box[1] - headgear.box[3]) / 2.0),
        )
        head_index += 1
        final.append(
            TeacherCandidate(
                candidate_id=f"H{head_index:02d}",
                kind="head",
                score=min(1.0, (headgear.score + head.score) / 2.0),
                box=union_box(headgear.box, head.box, width, height),
                source=f"non_safety_headgear_head_union({headgear.source};{head.source})",
                prompt=f"{headgear.prompt} + whole head",
                sam_refined=headgear.sam_refined or head.sam_refined,
                mask_score=headgear.mask_score,
            )
        )
    for index, shell in enumerate(shells, 1):
        shell_center = ((shell.box[0] + shell.box[2]) / 2.0, (shell.box[1] + shell.box[3]) / 2.0)
        head = min(
            head_candidates,
            key=lambda item: abs((item.box[0] + item.box[2]) / 2.0 - shell_center[0])
            + abs((item.box[1] + item.box[3]) / 2.0 - shell_center[1]),
        )
        final.append(
            TeacherCandidate(
                candidate_id=f"C{index:02d}",
                kind="helmet",
                score=min(1.0, (shell.score + head.score) / 2.0),
                box=union_box(shell.box, head.box, width, height),
                source=f"helmet_head_union({shell.source};{head.source})",
                prompt=f"{shell.prompt} + whole head",
                sam_refined=shell.sam_refined or head.sam_refined,
                mask_score=shell.mask_score,
            )
        )
    return final


def draw_candidate_overlay(
    image: np.ndarray,
    person: Box,
    candidates: Sequence[TeacherCandidate],
) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, person[:2], person[2:], (0, 255, 0), 3)
    cv2.putText(output, "TARGET PERSON", (person[0], max(22, person[1] + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
    colors = {"head": (0, 90, 255), "helmet": (255, 180, 0)}
    for item in candidates:
        color = colors.get(item.kind, (200, 200, 200))
        cv2.rectangle(output, item.box[:2], item.box[2:], color, 3)
        cv2.putText(output, item.candidate_id, (item.box[0], max(18, item.box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return output


def _data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("无法编码Qwen标注输入图")
    value = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{value}"


class QwenSemanticSelector:
    """Use the local OpenAI-compatible Qwen3-VL service as semantic authority."""

    VALID_STATES = {"helmet", "head", "person_only", "ignore", "no_person"}

    def __init__(self, config: Mapping[str, Any], project_root: Path) -> None:
        self.config = dict(config)
        self.server_url = str(self.config["server_url"])
        self.prompt = (project_root / str(self.config["prompt_path"])).read_text(encoding="utf-8").strip()
        self.schema = json.loads((project_root / str(self.config["schema_path"])).read_text(encoding="utf-8"))

    def review(
        self,
        image: np.ndarray,
        overlay: np.ndarray,
        person: Box,
        candidates: Sequence[TeacherCandidate],
        historical_review: Mapping[str, Any] | None = None,
    ) -> Tuple[SemanticDecision, Dict[str, Any]]:
        candidate_rows = [
            {
                "candidate_id": item.candidate_id,
                "kind": item.kind,
                "box": list(item.box),
                "source": item.source,
                "score": round(float(item.score), 4),
            }
            for item in candidates
        ]
        auxiliary = {
            "target_person_box": list(person),
            "candidates": candidate_rows,
            "historical_alarm_review": dict(historical_review or {}),
            "note": "历史告警结论仅作弱证据，必须以当前图像为准。",
        }
        content = [
            {"type": "text", "text": self.prompt},
            {"type": "text", "text": "图片1：无候选框的原始头肩裁剪图。"},
            {"type": "image_url", "image_url": {"url": _data_url(image)}},
            {"type": "text", "text": "图片2：绿色框是待判断人员，橙/蓝框为可选择候选框。"},
            {"type": "image_url", "image_url": {"url": _data_url(overlay)}},
            {"type": "text", "text": "候选信息：" + json.dumps(auxiliary, ensure_ascii=False, separators=(",", ":"))},
        ]
        messages = [{"role": "user", "content": content}]
        body = {
            "model": str(self.config["model"]),
            "temperature": 0,
            "max_tokens": int(self.config["max_tokens"]),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "helmet_annotation", "strict": True, "schema": self.schema},
            },
        }
        error: Exception | None = None
        retry_errors: List[str] = []
        for attempt in range(int(self.config["retries"]) + 1):
            request_body = dict(body)
            request_body["messages"] = messages
            encoded = json.dumps(request_body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            raw = ""
            try:
                request = urllib.request.Request(
                    self.server_url,
                    data=encoded,
                    headers={"Content-Type": "application/json; charset=utf-8"},
                    method="POST",
                )
                started = time.perf_counter()
                with urllib.request.urlopen(request, timeout=float(self.config["timeout_seconds"])) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                latency = time.perf_counter() - started
                raw = str(payload["choices"][0]["message"]["content"])
                value = json.loads(raw)
                decision = self._validate(value, candidates)
                return decision, {
                    "status": "completed",
                    "latency_seconds": round(latency, 3),
                    "attempt_count": attempt + 1,
                    "retry_errors": retry_errors,
                    "usage": payload.get("usage") or {},
                    "raw_response": raw,
                }
            except Exception as exc:
                error = exc
                retry_errors.append(str(exc))
                if attempt < int(self.config["retries"]):
                    if raw:
                        messages = messages + [
                            {"role": "assistant", "content": raw},
                            {
                                "role": "user",
                                "content": (
                                    "上次JSON未通过标注规则校验："
                                    f"{exc}。请重新检查候选kind与head_state，"
                                    "H候选只能用于head，C候选只能用于helmet；"
                                    "选择候选时direct_box必须为空。"
                                    "confidence必须表示语义判断把握，不能照抄候选score。"
                                    "只输出一份修正后的JSON。"
                                ),
                            },
                        ]
                    time.sleep(1.0)
        raise RuntimeError(f"Qwen语义标注失败：{error}")

    def _validate(
        self,
        value: Mapping[str, Any],
        candidates: Sequence[TeacherCandidate],
    ) -> SemanticDecision:
        state = str(value.get("head_state") or "").strip().lower()
        if state not in self.VALID_STATES:
            raise ValueError(f"非法head_state：{state}")
        is_person = bool(value.get("is_target_person"))
        selected = str(value.get("selected_candidate_id") or "").strip()
        direct_box = tuple(int(item) for item in (value.get("direct_box") or []))
        if len(direct_box) not in {0, 4}:
            raise ValueError("direct_box必须为空或包含4个坐标")
        if len(direct_box) == 4 and not (direct_box[0] < direct_box[2] and direct_box[1] < direct_box[3]):
            raise ValueError("direct_box坐标顺序非法")
        if selected:
            direct_box = ()
        candidate_map = {item.candidate_id: item for item in candidates}
        if state in {"head", "helmet"}:
            if not selected and len(direct_box) == 4:
                pass
            elif selected not in candidate_map:
                raise ValueError(f"选择了不存在的候选框：{selected}")
            elif candidate_map[selected].kind != state:
                raise ValueError(f"候选框类别不匹配：state={state} id={selected}")
        elif selected or direct_box:
            raise ValueError(f"状态{state}不应选择候选框")
        if not is_person and state != "no_person":
            raise ValueError("is_target_person=false时head_state必须为no_person")
        if is_person and state == "no_person":
            raise ValueError("is_target_person=true时head_state不能为no_person")
        return SemanticDecision(
            is_target_person=is_person,
            head_state=state,
            selected_candidate_id=selected,
            direct_box=direct_box,
            confidence=max(0.0, min(1.0, float(value.get("confidence", 0.0)))),
            helmet_color=str(value.get("helmet_color") or "uncertain"),
            wearing_state=str(value.get("wearing_state") or "unclear"),
            visibility_ratio=max(0.0, min(1.0, float(value.get("visibility_ratio", 0.0)))),
            truncated=bool(value.get("truncated")),
            reason=str(value.get("reason") or "").strip(),
        )


def fallback_decision(
    semantic_bucket: str,
    candidates: Sequence[TeacherCandidate],
    reason: str,
) -> SemanticDecision:
    # A failed semantic review must not turn an older alarm verdict into a new
    # head/helmet training label. Preserve only the independent person box.
    return SemanticDecision(
        is_target_person=True,
        head_state="person_only",
        selected_candidate_id="",
        direct_box=(),
        confidence=0.0,
        helmet_color="uncertain",
        wearing_state="unclear",
        visibility_ratio=0.0,
        truncated=False,
        reason=reason,
        source="historical_review_fallback_pending_manual_review",
    )
