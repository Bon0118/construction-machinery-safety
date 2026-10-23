"""Human review queue, annotation versions, and Web routes."""

import io
import json
import math
import sqlite3
import threading
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import config
from video_alarms import AlarmNotFoundError, AlarmValidationError, VideoAlarmStore
from video_channels import DEFAULT_DATABASE_PATH, TEMPLATES_DIR


ALLOWED_CLASSES = {"person", "head", "helmet"}
CLASS_IDS = {"person": 0, "head": 1, "helmet": 2}
WORKFLOW_STATUSES = {"draft", "completed", "conference"}
DECISIONS = {"confirmed", "corrected", "no_target", "uncertain", "ignored"}
TRAINING_DECISIONS = {"confirmed", "corrected", "no_target"}
SAFETY_HELMET_ALARM_TYPE = "安全帽检测"
REASON_CODES = {
    "missing_label",
    "wrong_class",
    "box_offset",
    "head_assignment_unclear",
    "poor_image_quality",
    "non_person_target",
    "other",
}


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ManualReviewStore:
    """Store manual annotation drafts without overwriting AI review evidence."""

    def __init__(
        self,
        alarm_store: VideoAlarmStore,
        database_path: Path = DEFAULT_DATABASE_PATH,
        dataset_output_dir: Optional[Path] = None,
    ) -> None:
        self.alarm_store = alarm_store
        self.database_path = Path(database_path)
        self.dataset_output_dir = Path(dataset_output_dir or (config.PROJECT_ROOT / "dataset_outputs"))
        self._lock = threading.RLock()
        self._teacher_annotation_index: Dict[str, Path] = {}
        self._initialize()
        self._refresh_teacher_annotation_index()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database_path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS manual_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alarm_id INTEGER NOT NULL UNIQUE,
                    workflow_status TEXT NOT NULL DEFAULT 'draft',
                    decision TEXT NOT NULL DEFAULT '',
                    reason_codes TEXT NOT NULL DEFAULT '[]',
                    comment TEXT NOT NULL DEFAULT '',
                    annotations TEXT NOT NULL DEFAULT '[]',
                    source_annotation_path TEXT NOT NULL DEFAULT '',
                    reviewer TEXT NOT NULL DEFAULT '本机审核员',
                    version INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (alarm_id) REFERENCES video_alarms(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS manual_review_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    manual_review_id INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    reason_codes TEXT NOT NULL DEFAULT '[]',
                    comment TEXT NOT NULL DEFAULT '',
                    annotations TEXT NOT NULL DEFAULT '[]',
                    reviewer TEXT NOT NULL DEFAULT '本机审核员',
                    created_at TEXT NOT NULL,
                    UNIQUE (manual_review_id, version),
                    FOREIGN KEY (manual_review_id) REFERENCES manual_reviews(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_manual_reviews_workflow ON manual_reviews(workflow_status, updated_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_manual_review_versions_review ON manual_review_versions(manual_review_id, version)"
            )

    @staticmethod
    def _json_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        try:
            parsed = json.loads(value or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    def _review_public(row: Optional[sqlite3.Row]) -> Dict[str, Any]:
        if row is None:
            return {
                "id": None,
                "workflow_status": "pending",
                "decision": "",
                "reason_codes": [],
                "comment": "",
                "annotations": [],
                "source_annotation_path": "",
                "reviewer": "本机审核员",
                "version": 0,
                "created_at": None,
                "updated_at": None,
                "completed_at": None,
            }
        data = dict(row)
        return {
            "id": int(data["id"]),
            "workflow_status": data.get("workflow_status") or "draft",
            "decision": data.get("decision") or "",
            "reason_codes": ManualReviewStore._json_list(data.get("reason_codes")),
            "comment": data.get("comment") or "",
            "annotations": ManualReviewStore._json_list(data.get("annotations")),
            "source_annotation_path": data.get("source_annotation_path") or "",
            "reviewer": data.get("reviewer") or "本机审核员",
            "version": int(data.get("version") or 0),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "completed_at": data.get("completed_at"),
        }

    def _alarm_row(self, alarm_id: int) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM video_alarms WHERE id = ?", (int(alarm_id),)).fetchone()
        if row is None:
            raise AlarmNotFoundError("人工复核样本不存在")
        if str(row["alarm_type"] or "") != SAFETY_HELMET_ALARM_TYPE:
            raise AlarmValidationError("当前人工标注规范仅支持安全帽检测（person/head/helmet）")
        if str(row["event_id"] or "") not in self._teacher_annotation_index:
            raise AlarmValidationError("当前人工复核模块仅处理具有教师模型候选预标注的样本")
        return row

    def _manual_row(self, alarm_id: int) -> Optional[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute("SELECT * FROM manual_reviews WHERE alarm_id = ?", (int(alarm_id),)).fetchone()

    def _event_dir(self, alarm_row: sqlite3.Row) -> Path:
        event_dir = Path(str(alarm_row["image_path"])).resolve().parent
        output_root = self.alarm_store.output_dir.resolve()
        if event_dir.parent != output_root:
            raise AlarmNotFoundError("告警事件目录无效")
        return event_dir

    def image_path(self, alarm_id: int, kind: str = "original") -> Path:
        alarm_row = self._alarm_row(alarm_id)
        event_dir = self._event_dir(alarm_row)
        normalized = str(kind or "original").strip().lower()
        candidates = [event_dir / "original.jpg", Path(str(alarm_row["image_path"]))]
        if normalized == "alarm":
            candidates.reverse()
        for path in candidates:
            resolved = path.resolve()
            if resolved.parent == event_dir and resolved.is_file():
                return resolved
        raise AlarmNotFoundError("复核图片不存在")

    def _refresh_teacher_annotation_index(self) -> None:
        """Index teacher JSON names once so opening a live alarm stays responsive."""
        if not self.dataset_output_dir.is_dir():
            self._teacher_annotation_index = {}
            return
        try:
            runs = sorted(
                (path for path in self.dataset_output_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            self._teacher_annotation_index = {}
            return
        index: Dict[str, Path] = {}
        for run_dir in runs:
            annotations_dir = run_dir / "annotations"
            if not annotations_dir.is_dir():
                continue
            for path in annotations_dir.glob("*.json"):
                stem = path.stem
                marker = stem.find("_event_")
                if marker >= 0:
                    event_id = stem[marker + 1 :]
                elif stem.startswith("event_"):
                    event_id = stem
                else:
                    continue
                if event_id not in index:
                    index[event_id] = path.resolve()
        self._teacher_annotation_index = index

    def _find_teacher_annotation(self, event_id: str) -> Optional[Path]:
        return self._teacher_annotation_index.get(str(event_id))

    @staticmethod
    def _annotation_public(raw: Dict[str, Any], index: int, prefix: str) -> Optional[Dict[str, Any]]:
        class_name = str(raw.get("class_name") or "").strip().lower()
        box = raw.get("box")
        if class_name not in ALLOWED_CLASSES or not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        try:
            coordinates = [round(float(value), 2) for value in box]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in coordinates):
            return None
        return {
            "id": str(raw.get("id") or f"{prefix}-{index}"),
            "class_name": class_name,
            "box": coordinates,
            "truncated": bool(raw.get("truncated", False)),
            "color": str(raw.get("color") or raw.get("helmet_color") or "")[:32],
            "confidence": raw.get("confidence"),
            "source": str(raw.get("source") or prefix),
        }

    @staticmethod
    def _public_box(raw: Any) -> Optional[List[float]]:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return None
        try:
            box = [round(float(value), 2) for value in raw]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
            return None
        return box

    @staticmethod
    def _crop_box_to_original(box: List[float], transform: Dict[str, Any]) -> Optional[List[float]]:
        crop_box = ManualReviewStore._public_box(transform.get("crop_box_original"))
        output_size = transform.get("output_size")
        if crop_box is None or not isinstance(output_size, (list, tuple)) or len(output_size) != 2:
            return None
        try:
            output_width, output_height = float(output_size[0]), float(output_size[1])
        except (TypeError, ValueError):
            return None
        if output_width <= 0 or output_height <= 0:
            return None
        scale_x = (crop_box[2] - crop_box[0]) / output_width
        scale_y = (crop_box[3] - crop_box[1]) / output_height
        return [
            round(crop_box[0] + box[0] * scale_x, 2),
            round(crop_box[1] + box[1] * scale_y, 2),
            round(crop_box[0] + box[2] * scale_x, 2),
            round(crop_box[1] + box[3] * scale_y, 2),
        ]

    @staticmethod
    def _boxes_related(person_box: List[float], target_box: List[float]) -> bool:
        center_x = (target_box[0] + target_box[2]) / 2.0
        center_y = (target_box[1] + target_box[3]) / 2.0
        center_inside = (
            person_box[0] <= center_x <= person_box[2]
            and person_box[1] - (person_box[3] - person_box[1]) * 0.15 <= center_y <= person_box[3]
        )
        return center_inside or ManualReviewStore._overlap_over_smaller(person_box, target_box) >= 0.25

    def _person_head_statuses(
        self,
        teacher_path: Path,
        teacher_data: Dict[str, Any],
        full_labels: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        crop_ids = teacher_data.get("crop_ids") if isinstance(teacher_data.get("crop_ids"), list) else []
        crop_annotations_dir = teacher_path.parent.parent / "crops" / "annotations"
        full_targets = [
            item for item in full_labels
            if str(item.get("class_name") or "").lower() in {"head", "helmet"}
            and self._public_box(item.get("box")) is not None
        ]
        statuses: List[Dict[str, Any]] = []
        for raw_crop_id in crop_ids:
            crop_id = str(raw_crop_id or "").strip()
            if not crop_id or Path(crop_id).name != crop_id:
                continue
            crop_path = crop_annotations_dir / f"{crop_id}.json"
            try:
                crop_data = json.loads(crop_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            transform = crop_data.get("crop_transform") if isinstance(crop_data.get("crop_transform"), dict) else {}
            person_box = self._public_box(transform.get("source_person_box_original"))
            if person_box is None:
                person_box = self._public_box((crop_data.get("independent_person_detection") or {}).get("box"))
            if person_box is None:
                continue
            crop_labels = crop_data.get("labels") if isinstance(crop_data.get("labels"), list) else []
            target = next(
                (item for item in crop_labels if str(item.get("class_name") or "").lower() in {"head", "helmet"}),
                None,
            )
            decision = crop_data.get("qwen_semantic_decision") if isinstance(crop_data.get("qwen_semantic_decision"), dict) else {}
            runtime = crop_data.get("qwen_runtime") if isinstance(crop_data.get("qwen_runtime"), dict) else {}
            issues = [str(item) for item in crop_data.get("manual_review_reasons", []) if item]
            target_box = self._public_box(target.get("box")) if target else None
            if target_box is None and str(decision.get("head_state") or "") in {"head", "helmet"}:
                selected_id = str(decision.get("selected_candidate_id") or "")
                candidates = crop_data.get("label_candidates") if isinstance(crop_data.get("label_candidates"), list) else []
                selected = next((item for item in candidates if str(item.get("candidate_id") or "") == selected_id), None)
                target_box = self._public_box(selected.get("box")) if selected else None
            mapped_target = self._crop_box_to_original(target_box, transform) if target_box else None
            training_short_edge = round(min(target_box[2] - target_box[0], target_box[3] - target_box[1]), 1) if target_box else None
            original_short_edge = round(min(mapped_target[2] - mapped_target[0], mapped_target[3] - mapped_target[1]), 1) if mapped_target else None
            associated_full = next(
                (
                    item for item in full_targets
                    if self._boxes_related(person_box, self._public_box(item.get("box")) or [0, 0, 0, 0])
                ),
                None,
            )
            semantic_state = str(decision.get("head_state") or "unknown")
            reason = str(decision.get("reason") or runtime.get("error") or "").strip()
            if associated_full is not None:
                target_class = str(associated_full.get("class_name") or "").lower()
                code, label, short_label = "labeled", f"已标注 {target_class}", f"已标 {target_class}"
            elif target is not None or any("below_20px_at_training_size" in item for item in issues):
                target_class = str((target or {}).get("class_name") or semantic_state or "head/helmet").lower()
                code, label, short_label = "filtered_small", "小目标过滤", "小目标过滤"
                reason = (
                    f"头肩裁剪中已定位 {target_class}，但映射到原图并按统一训练尺寸检查后，"
                    "短边不足 20 像素或映射框无效，因此原图候选标注未保留。"
                )
            elif "qwen_review_failed" in issues or "qwen_candidate_selection_invalid" in issues or str(runtime.get("status") or "") in {"fallback", "failed", "error"}:
                code, label, short_label = "suspected_missing", "疑似漏标", "疑似漏标"
                reason = reason or "教师模型处理失败或候选框选择无效，需要人工检查并补标。"
            elif "qwen_ignore" in issues or semantic_state == "ignore":
                code, label, short_label = "ignored_quality", "主动忽略", "主动忽略"
                reason = reason or "头部模糊、遮挡或不可见，教师模型按规则主动忽略。"
            elif "qwen_person_only" in issues or semantic_state == "person_only":
                code, label, short_label = "model_uncertain", "模型未确定", "模型未确定"
                reason = reason or "教师模型只确认了人员，未能可靠判断 head/helmet。"
            else:
                code, label, short_label = "suspected_missing", "疑似漏标", "疑似漏标"
                reason = reason or "存在人员框但没有 head/helmet，且没有明确的过滤或忽略原因。"
            statuses.append(
                {
                    "person_box": person_box,
                    "code": code,
                    "label": label,
                    "short_label": short_label,
                    "reason": reason,
                    "crop_id": crop_id,
                    "semantic_state": semantic_state,
                    "training_short_edge": training_short_edge,
                    "original_short_edge": original_short_edge,
                    "minimum_short_edge": 20,
                    "qwen_confidence": decision.get("confidence"),
                    "reason_codes": issues,
                }
            )
        return statuses

    def _attach_person_head_statuses(
        self,
        annotations: List[Dict[str, Any]],
        statuses: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result = [dict(item) for item in annotations]
        for item in result:
            if item.get("class_name") != "person":
                continue
            box = self._public_box(item.get("box"))
            if box is None:
                continue
            matches = sorted(
                (
                    (self._overlap_over_smaller(box, status["person_box"]), status)
                    for status in statuses
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            if matches and matches[0][0] >= 0.5:
                item["head_status"] = {key: value for key, value in matches[0][1].items() if key != "person_box"}
        return result

    def _candidate_annotations(self, alarm_row: sqlite3.Row) -> Tuple[List[Dict[str, Any]], str]:
        event_id = str(alarm_row["event_id"])
        teacher_path = self._find_teacher_annotation(event_id)
        if teacher_path is None:
            raise AlarmValidationError("教师模型候选预标注文件不存在")
        try:
            data = json.loads(teacher_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise AlarmValidationError(f"教师模型候选预标注文件无法读取：{exc}") from exc
        labels = data.get("labels") if isinstance(data.get("labels"), list) else []
        result = [self._annotation_public(item, index + 1, "teacher") for index, item in enumerate(labels)]
        public_labels = [item for item in result if item is not None]
        statuses = self._person_head_statuses(teacher_path, data, labels)
        public_labels = self._attach_person_head_statuses(public_labels, statuses)
        for item in public_labels:
            if item.get("class_name") == "person" and not isinstance(item.get("head_status"), dict):
                item["head_status"] = {
                    "code": "unavailable",
                    "label": "原因不可追溯",
                    "short_label": "原因缺失",
                    "reason": "该教师预标注运行没有保存与此人员对应的裁剪级状态，不能判断是规则过滤还是漏标，需要人工确认。",
                    "crop_id": "",
                    "semantic_state": "unknown",
                    "training_short_edge": None,
                    "original_short_edge": None,
                    "minimum_short_edge": 20,
                    "qwen_confidence": None,
                    "reason_codes": [],
                }
        return public_labels, str(teacher_path)

    @staticmethod
    def _image_size(path: Path) -> Tuple[int, int]:
        try:
            from PIL import Image

            with Image.open(str(path)) as image:
                return int(image.width), int(image.height)
        except Exception as exc:
            raise AlarmValidationError(f"无法读取复核图片尺寸：{exc}") from exc

    @staticmethod
    def _normalize_reasons(values: Any) -> List[str]:
        if not isinstance(values, list):
            raise AlarmValidationError("问题归因必须是数组")
        result = []
        for raw in values:
            code = str(raw or "").strip()
            if code not in REASON_CODES:
                raise AlarmValidationError(f"不支持的问题归因：{code}")
            if code not in result:
                result.append(code)
        return result

    @staticmethod
    def _overlap_over_smaller(first: List[float], second: List[float]) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        smaller = min((first[2] - first[0]) * (first[3] - first[1]), (second[2] - second[0]) * (second[3] - second[1]))
        return intersection / smaller if smaller > 0 else 0.0

    def _normalize_annotations(self, values: Any, width: int, height: int) -> List[Dict[str, Any]]:
        if not isinstance(values, list):
            raise AlarmValidationError("annotations 必须是数组")
        result: List[Dict[str, Any]] = []
        for index, raw in enumerate(values):
            if not isinstance(raw, dict):
                raise AlarmValidationError("标注对象格式无效")
            class_name = str(raw.get("class_name") or "").strip().lower()
            if class_name not in ALLOWED_CLASSES:
                raise AlarmValidationError("标注类别只能是 person、head 或 helmet")
            box = raw.get("box")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise AlarmValidationError("标注框必须包含 x1、y1、x2、y2")
            try:
                x1, y1, x2, y2 = [float(value) for value in box]
            except (TypeError, ValueError) as exc:
                raise AlarmValidationError("标注框坐标必须是数字") from exc
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                raise AlarmValidationError("标注框坐标无效")
            x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
            y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
            if x2 - x1 < 2 or y2 - y1 < 2:
                raise AlarmValidationError("标注框尺寸过小")
            if class_name in {"head", "helmet"} and min(x2 - x1, y2 - y1) < 20:
                raise AlarmValidationError("head/helmet 短边小于 20 像素，不应标注")
            result.append(
                {
                    "id": str(raw.get("id") or f"human-{index + 1}"),
                    "class_name": class_name,
                    "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "truncated": bool(raw.get("truncated", False)),
                    "color": str(raw.get("color") or "")[:32] if class_name == "helmet" else "",
                    "confidence": None,
                    "source": "human",
                }
            )

        heads = [item for item in result if item["class_name"] == "head"]
        helmets = [item for item in result if item["class_name"] == "helmet"]
        for head in heads:
            for helmet in helmets:
                if self._overlap_over_smaller(head["box"], helmet["box"]) >= 0.7:
                    raise AlarmValidationError("同一人头不能同时标注为 head 和 helmet")
        return result

    def list_queue(
        self,
        ai_status: str = "",
        human_status: str = "",
        task_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        alarm_date: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, min(100, int(page_size)))
        teacher_event_ids = sorted(self._teacher_annotation_index)
        if not teacher_event_ids:
            return {
                "items": [],
                "total": 0,
                "page": 1,
                "page_size": max(1, min(100, int(page_size))),
                "pages": 1,
                "metrics": {"pending": 0, "draft": 0, "conference": 0, "completed_today": 0, "correction_rate": 0.0},
            }
        teacher_placeholders = ",".join("?" for _ in teacher_event_ids)
        where: List[str] = ["a.alarm_type = ?", f"a.event_id IN ({teacher_placeholders})"]
        values: List[Any] = [SAFETY_HELMET_ALARM_TYPE, *teacher_event_ids]
        ai_text = str(ai_status or "").strip().lower()
        if ai_text:
            if ai_text not in {"confirmed", "rejected", "uncertain", "pending", "reviewing"}:
                raise AlarmValidationError("AI 复核结果无效")
            where.append("a.review_status = ?")
            values.append(ai_text)
        human_text = str(human_status or "").strip().lower()
        if human_text:
            if human_text == "pending":
                where.append("mr.id IS NULL")
            elif human_text in WORKFLOW_STATUSES:
                where.append("mr.workflow_status = ?")
                values.append(human_text)
            else:
                raise AlarmValidationError("人工复核状态无效")
        if task_id is not None:
            where.append("a.task_id = ?")
            values.append(int(task_id))
        if channel_id is not None:
            where.append("a.channel_id = ?")
            values.append(int(channel_id))
        date_text = str(alarm_date or "").strip()
        if date_text:
            try:
                datetime.strptime(date_text, "%Y-%m-%d")
            except ValueError as exc:
                raise AlarmValidationError("告警日期格式必须是 YYYY-MM-DD") from exc
            where.append("substr(a.alarm_time, 1, 10) = ?")
            values.append(date_text)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        select = """
            SELECT a.*, mr.id AS human_review_id,
                   COALESCE(mr.workflow_status, 'pending') AS human_workflow_status,
                   COALESCE(mr.decision, '') AS human_decision,
                   COALESCE(mr.annotations, '[]') AS human_annotations,
                   COALESCE(mr.updated_at, '') AS human_updated_at
            FROM video_alarms a LEFT JOIN manual_reviews mr ON mr.alarm_id = a.id
        """
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM ({select} {clause})", values).fetchone()[0])
            pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, pages)
            rows = connection.execute(
                f"{select} {clause} ORDER BY CASE a.review_status WHEN 'rejected' THEN 0 WHEN 'uncertain' THEN 1 ELSE 2 END, a.alarm_time DESC, a.id DESC LIMIT ? OFFSET ?",
                [*values, page_size, (page - 1) * page_size],
            ).fetchall()
            metrics = connection.execute(
                f"""
                SELECT
                    SUM(CASE WHEN mr.id IS NULL THEN 1 ELSE 0 END) AS pending_count,
                    SUM(CASE WHEN mr.workflow_status = 'draft' THEN 1 ELSE 0 END) AS draft_count,
                    SUM(CASE WHEN mr.workflow_status = 'conference' THEN 1 ELSE 0 END) AS conference_count,
                    SUM(CASE WHEN mr.workflow_status = 'completed' AND substr(mr.completed_at, 1, 10) = ? THEN 1 ELSE 0 END) AS completed_today,
                    SUM(CASE WHEN mr.decision = 'corrected' THEN 1 ELSE 0 END) AS corrected_count,
                    SUM(CASE WHEN mr.workflow_status = 'completed' THEN 1 ELSE 0 END) AS completed_count
                FROM video_alarms a LEFT JOIN manual_reviews mr ON mr.alarm_id = a.id
                WHERE a.alarm_type = ? AND a.event_id IN ({teacher_placeholders})
                """,
                (datetime.now().astimezone().date().isoformat(), SAFETY_HELMET_ALARM_TYPE, *teacher_event_ids),
            ).fetchone()
        items = []
        for row in rows:
            alarm = self.alarm_store._row_to_public(row)
            annotations = self._json_list(row["human_annotations"])
            alarm.update(
                {
                    "human_review_id": int(row["human_review_id"]) if row["human_review_id"] is not None else None,
                    "human_workflow_status": row["human_workflow_status"],
                    "human_decision": row["human_decision"],
                    "human_annotation_count": len(annotations),
                    "human_updated_at": row["human_updated_at"],
                }
            )
            items.append(alarm)
        completed = int(metrics["completed_count"] or 0)
        corrected = int(metrics["corrected_count"] or 0)
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
            "metrics": {
                "pending": int(metrics["pending_count"] or 0),
                "draft": int(metrics["draft_count"] or 0),
                "conference": int(metrics["conference_count"] or 0),
                "completed_today": int(metrics["completed_today"] or 0),
                "correction_rate": round(corrected / completed, 4) if completed else 0.0,
            },
        }

    def detail(self, alarm_id: int) -> Dict[str, Any]:
        alarm_row = self._alarm_row(alarm_id)
        manual_row = self._manual_row(alarm_id)
        original_path = self.image_path(alarm_id, "original")
        width, height = self._image_size(original_path)
        candidates, source_path = self._candidate_annotations(alarm_row)
        review = self._review_public(manual_row)
        annotations = review["annotations"] if manual_row is not None else candidates
        if manual_row is not None:
            candidate_statuses = [
                {"person_box": item["box"], **item["head_status"]}
                for item in candidates
                if item.get("class_name") == "person" and isinstance(item.get("head_status"), dict)
            ]
            annotations = self._attach_person_head_statuses(annotations, candidate_statuses)
        alarm = self.alarm_store._row_to_public(alarm_row)
        with self._connect() as connection:
            version_rows = []
            if manual_row is not None:
                version_rows = connection.execute(
                    "SELECT version, decision, reviewer, created_at FROM manual_review_versions WHERE manual_review_id = ? ORDER BY version DESC",
                    (int(manual_row["id"]),),
                ).fetchall()
        return {
            "alarm": alarm,
            "review": review,
            "candidate_annotations": candidates,
            "annotations": annotations,
            "candidate_source_path": source_path,
            "image": {
                "width": width,
                "height": height,
                "original_url": f"/api/manual-reviews/{int(alarm_id)}/image?kind=original",
                "alarm_url": f"/api/manual-reviews/{int(alarm_id)}/image?kind=alarm",
            },
            "versions": [dict(row) for row in version_rows],
        }

    def _write_sidecar(self, alarm_row: sqlite3.Row, snapshot: Dict[str, Any], submit: bool) -> None:
        event_dir = self._event_dir(alarm_row)
        review_dir = event_dir / "human_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        if submit:
            version_path = review_dir / f"annotation_v{int(snapshot['version']):03d}.json"
            version_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            current_path = review_dir / "current.json"
        else:
            current_path = review_dir / "draft.json"
        temporary = current_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(current_path)

    def save(self, alarm_id: int, payload: Dict[str, Any], submit: bool = False) -> Dict[str, Any]:
        alarm_row = self._alarm_row(alarm_id)
        width, height = self._image_size(self.image_path(alarm_id, "original"))
        annotations = self._normalize_annotations(payload.get("annotations", []), width, height)
        reasons = self._normalize_reasons(payload.get("reason_codes", []))
        comment = str(payload.get("comment") or "").strip()[:4000]
        reviewer = str(payload.get("reviewer") or "本机审核员").strip()[:80] or "本机审核员"
        decision = str(payload.get("decision") or "").strip().lower()
        workflow_status = "draft"
        completed_at: Optional[str] = None
        if submit:
            if decision not in DECISIONS:
                raise AlarmValidationError("请选择有效的人工结论")
            if decision == "no_target":
                annotations = []
            elif decision in {"confirmed", "corrected"} and not annotations:
                raise AlarmValidationError("确认有效样本前至少需要一个人工标注框")
            if decision in {"corrected", "uncertain", "ignored"} and not (reasons or comment):
                raise AlarmValidationError("该结论需要填写问题归因或人工审核意见")
            workflow_status = "conference" if decision == "uncertain" else "completed"
            completed_at = now_text()
        elif decision and decision not in DECISIONS:
            raise AlarmValidationError("人工结论无效")

        candidates, source_path = self._candidate_annotations(alarm_row)
        timestamp = now_text()
        with self._lock, self._connect() as connection:
            existing = connection.execute("SELECT * FROM manual_reviews WHERE alarm_id = ?", (int(alarm_id),)).fetchone()
            version = int(existing["version"] or 0) if existing is not None else 0
            if submit:
                version += 1
            created_at = str(existing["created_at"]) if existing is not None else timestamp
            if existing is None:
                cursor = connection.execute(
                    """
                    INSERT INTO manual_reviews (
                        alarm_id, workflow_status, decision, reason_codes, comment, annotations,
                        source_annotation_path, reviewer, version, created_at, updated_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(alarm_id), workflow_status, decision, json.dumps(reasons, ensure_ascii=False), comment,
                        json.dumps(annotations, ensure_ascii=False), source_path, reviewer, version,
                        created_at, timestamp, completed_at,
                    ),
                )
                manual_review_id = int(cursor.lastrowid)
            else:
                manual_review_id = int(existing["id"])
                connection.execute(
                    """
                    UPDATE manual_reviews SET workflow_status = ?, decision = ?, reason_codes = ?,
                        comment = ?, annotations = ?, source_annotation_path = ?, reviewer = ?,
                        version = ?, updated_at = ?, completed_at = ? WHERE id = ?
                    """,
                    (
                        workflow_status, decision, json.dumps(reasons, ensure_ascii=False), comment,
                        json.dumps(annotations, ensure_ascii=False), source_path, reviewer, version,
                        timestamp, completed_at, manual_review_id,
                    ),
                )
            if submit:
                connection.execute(
                    """
                    INSERT INTO manual_review_versions (
                        manual_review_id, version, decision, reason_codes, comment,
                        annotations, reviewer, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        manual_review_id, version, decision, json.dumps(reasons, ensure_ascii=False),
                        comment, json.dumps(annotations, ensure_ascii=False), reviewer, timestamp,
                    ),
                )

        snapshot = {
            "schema_version": "manual-review-v1",
            "alarm_id": int(alarm_id),
            "event_id": str(alarm_row["event_id"]),
            "ai_review": {
                "status": str(alarm_row["review_status"] or "uncertain"),
                "confidence": alarm_row["review_confidence"],
                "summary": str(alarm_row["review_summary"] or ""),
                "reason_codes": self._json_list(alarm_row["review_reason_codes"]),
            },
            "workflow_status": workflow_status,
            "decision": decision,
            "reason_codes": reasons,
            "comment": comment,
            "annotations": annotations,
            "candidate_annotations": candidates,
            "candidate_source_path": source_path,
            "image_width": width,
            "image_height": height,
            "reviewer": reviewer,
            "version": version,
            "updated_at": timestamp,
            "training_eligibility": (
                "positive" if decision in {"confirmed", "corrected"}
                else "negative" if decision == "no_target"
                else "excluded"
            ),
        }
        self._write_sidecar(alarm_row, snapshot, submit=submit)
        return self.detail(alarm_id)

    def list_history(
        self,
        decision: str = "",
        reason_code: str = "",
        review_date: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, min(100, int(page_size)))
        teacher_event_ids = sorted(self._teacher_annotation_index)
        if not teacher_event_ids:
            return {"items": [], "total": 0, "page": 1, "page_size": max(1, min(100, int(page_size))), "pages": 1}
        teacher_placeholders = ",".join("?" for _ in teacher_event_ids)
        where = ["mr.workflow_status IN ('completed', 'conference')", f"a.event_id IN ({teacher_placeholders})"]
        values: List[Any] = [*teacher_event_ids]
        decision_text = str(decision or "").strip().lower()
        if decision_text:
            if decision_text not in DECISIONS:
                raise AlarmValidationError("人工结论无效")
            where.append("mr.decision = ?")
            values.append(decision_text)
        reason_text = str(reason_code or "").strip()
        if reason_text:
            if reason_text not in REASON_CODES:
                raise AlarmValidationError("问题归因无效")
            where.append("mr.reason_codes LIKE ?")
            values.append(f'%"{reason_text}"%')
        date_text = str(review_date or "").strip()
        if date_text:
            try:
                datetime.strptime(date_text, "%Y-%m-%d")
            except ValueError as exc:
                raise AlarmValidationError("审核日期格式必须是 YYYY-MM-DD") from exc
            where.append("substr(mr.completed_at, 1, 10) = ?")
            values.append(date_text)
        clause = f"WHERE {' AND '.join(where)}"
        base = """
            FROM manual_reviews mr JOIN video_alarms a ON a.id = mr.alarm_id
        """
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) {base} {clause}", values).fetchone()[0])
            pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, pages)
            rows = connection.execute(
                f"""
                SELECT mr.*, a.event_id, a.task_code, a.channel_name, a.alarm_time,
                       a.review_status AS ai_review_status, a.review_summary AS ai_review_summary
                {base} {clause}
                ORDER BY mr.completed_at DESC, mr.id DESC LIMIT ? OFFSET ?
                """,
                [*values, page_size, (page - 1) * page_size],
            ).fetchall()
        items = []
        for row in rows:
            annotations = self._json_list(row["annotations"])
            items.append(
                {
                    "id": int(row["id"]),
                    "alarm_id": int(row["alarm_id"]),
                    "event_id": row["event_id"],
                    "task_code": row["task_code"],
                    "channel_name": row["channel_name"],
                    "alarm_time": row["alarm_time"],
                    "ai_review_status": row["ai_review_status"],
                    "ai_review_summary": row["ai_review_summary"],
                    "workflow_status": row["workflow_status"],
                    "decision": row["decision"],
                    "reason_codes": self._json_list(row["reason_codes"]),
                    "comment": row["comment"],
                    "annotation_count": len(annotations),
                    "annotation_counts": {
                        class_name: sum(1 for item in annotations if item.get("class_name") == class_name)
                        for class_name in ("person", "head", "helmet")
                    },
                    "reviewer": row["reviewer"],
                    "version": int(row["version"] or 0),
                    "completed_at": row["completed_at"],
                    "training_eligibility": (
                        "positive" if row["decision"] in {"confirmed", "corrected"}
                        else "negative" if row["decision"] == "no_target"
                        else "excluded"
                    ),
                }
            )
        return {"items": items, "total": total, "page": page, "page_size": page_size, "pages": pages}

    def export_archive(self) -> Tuple[bytes, int]:
        teacher_event_ids = sorted(self._teacher_annotation_index)
        if not teacher_event_ids:
            teacher_event_ids = ["__no_teacher_samples__"]
        teacher_placeholders = ",".join("?" for _ in teacher_event_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT mr.*, a.event_id, a.image_path
                FROM manual_reviews mr JOIN video_alarms a ON a.id = mr.alarm_id
                WHERE mr.workflow_status = 'completed' AND mr.decision IN ('confirmed', 'corrected', 'no_target')
                  AND a.event_id IN ({teacher_placeholders})
                ORDER BY mr.completed_at, mr.id
                """,
                teacher_event_ids,
            ).fetchall()
        buffer = io.BytesIO()
        manifest: List[str] = []
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("classes.txt", "person\nhead\nhelmet\n")
            archive.writestr(
                "data.yaml",
                "path: .\ntrain: images\nval: images\nnc: 3\nnames: [person, head, helmet]\n",
            )
            for row in rows:
                alarm_id = int(row["alarm_id"])
                image_path = self.image_path(alarm_id, "original")
                width, height = self._image_size(image_path)
                event_id = str(row["event_id"])
                suffix = image_path.suffix.lower() or ".jpg"
                archive.write(str(image_path), f"images/{event_id}{suffix}")
                annotations = self._json_list(row["annotations"])
                yolo_lines = []
                for item in annotations:
                    class_name = item.get("class_name")
                    if class_name not in CLASS_IDS:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in item["box"]]
                    center_x = ((x1 + x2) / 2.0) / width
                    center_y = ((y1 + y2) / 2.0) / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    yolo_lines.append(
                        f"{CLASS_IDS[class_name]} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
                    )
                archive.writestr(f"labels/{event_id}.txt", "\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
                metadata = {
                    "event_id": event_id,
                    "decision": row["decision"],
                    "reason_codes": self._json_list(row["reason_codes"]),
                    "comment": row["comment"],
                    "annotations": annotations,
                    "reviewer": row["reviewer"],
                    "version": int(row["version"] or 0),
                    "completed_at": row["completed_at"],
                }
                archive.writestr(
                    f"metadata/{event_id}.json",
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                )
                manifest.append(json.dumps(metadata, ensure_ascii=False))
            archive.writestr("manifest.jsonl", "\n".join(manifest) + ("\n" if manifest else ""))
        return buffer.getvalue(), len(rows)


def register_manual_review_routes(app: Any, store: ManualReviewStore) -> None:
    """Attach the manual-review module and APIs to FastAPI."""
    from fastapi import HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, Response
    from fastapi.templating import Jinja2Templates

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def handle_error(exc: BaseException) -> None:
        if isinstance(exc, AlarmNotFoundError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, AlarmValidationError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise exc

    @app.get("/manual-reviews", response_class=HTMLResponse, include_in_schema=False)
    def manual_reviews_page(request: Request) -> Any:
        return templates.TemplateResponse(
            request=request,
            name="manual_reviews.html",
            context={"page_title": "人工复核"},
        )

    @app.get("/api/manual-reviews")
    def list_manual_review_queue(
        ai_status: str = "",
        human_status: str = "",
        task_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        alarm_date: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        try:
            return store.list_queue(
                ai_status=ai_status,
                human_status=human_status,
                task_id=task_id,
                channel_id=channel_id,
                alarm_date=alarm_date,
                page=page,
                page_size=page_size,
            )
        except (AlarmNotFoundError, AlarmValidationError) as exc:
            handle_error(exc)
            raise AssertionError("unreachable")

    @app.get("/api/manual-reviews/export")
    def export_manual_reviews() -> Response:
        archive, count = store.export_archive()
        filename = f"manual_review_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        return Response(
            content=archive,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Exported-Samples": str(count),
            },
        )

    @app.get("/api/manual-reviews/history")
    def list_manual_review_history(
        decision: str = "",
        reason_code: str = "",
        review_date: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        try:
            return store.list_history(
                decision=decision,
                reason_code=reason_code,
                review_date=review_date,
                page=page,
                page_size=page_size,
            )
        except (AlarmNotFoundError, AlarmValidationError) as exc:
            handle_error(exc)
            raise AssertionError("unreachable")

    @app.get("/api/manual-reviews/{alarm_id}")
    def manual_review_detail(alarm_id: int) -> Dict[str, Any]:
        try:
            return store.detail(alarm_id)
        except (AlarmNotFoundError, AlarmValidationError) as exc:
            handle_error(exc)
            raise AssertionError("unreachable")

    @app.get("/api/manual-reviews/{alarm_id}/image", include_in_schema=False)
    def manual_review_image(alarm_id: int, kind: str = "original") -> FileResponse:
        try:
            path = store.image_path(alarm_id, kind)
        except (AlarmNotFoundError, AlarmValidationError) as exc:
            handle_error(exc)
            raise AssertionError("unreachable")
        return FileResponse(path=str(path))

    @app.put("/api/manual-reviews/{alarm_id}/draft")
    def save_manual_review_draft(alarm_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return store.save(alarm_id, payload, submit=False)
        except (AlarmNotFoundError, AlarmValidationError) as exc:
            handle_error(exc)
            raise AssertionError("unreachable")

    @app.post("/api/manual-reviews/{alarm_id}/submit")
    def submit_manual_review(alarm_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return store.save(alarm_id, payload, submit=True)
        except (AlarmNotFoundError, AlarmValidationError) as exc:
            handle_error(exc)
            raise AssertionError("unreachable")
