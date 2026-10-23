"""Video-task persistence, validation, and Web routes."""

import json
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlsplit

import push_alarm
from algorithms import algorithm_registry
from algorithms.machinery_hazard.defaults import (
    ALARM_SAVE_COOLDOWN_SECONDS,
    ALGORITHM_CODE as DEFAULT_ALGORITHM_CODE,
    CONFIDENCE_THRESHOLD as MACHINERY_CONFIDENCE_THRESHOLD,
    OVERLAP_RATIO_THRESHOLD,
    PERSON_CONFIDENCE_THRESHOLD as MACHINERY_PERSON_CONFIDENCE_THRESHOLD,
    RISK_CONSECUTIVE_FRAMES,
)
from algorithms.yolov5_objects.defaults import OBJECT_ALGORITHM_DEFINITIONS
from video_channels import DEFAULT_DATABASE_PATH, TEMPLATES_DIR


TASK_STATUS_RUNNING = "running"
TASK_STATUS_STOPPED = "stopped"
TASK_STATUSES = {TASK_STATUS_RUNNING, TASK_STATUS_STOPPED}
DEFAULT_REPORT_URL = str(getattr(push_alarm, "API_URL", "") or "").strip()
REPORT_MAPPING_KEYS = ("task_code", "camera_id", "event_type", "event_time", "image")
DEFAULT_RUNTIME_POLICY = {
    "schedule_enabled": False,
    "schedule_start": "00:00",
    "schedule_end": "23:59",
    "event_wakeup_enabled": False,
    "event_wakeup_seconds": 300,
    "main_stream_confirmation_enabled": False,
    "small_target_area_ratio": 0.02,
    "confirmation_timeout_seconds": 10,
}


def _number_parameter(
    key: str,
    name: str,
    default: float,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
    step: float = 0.01,
    unit: str = "",
    help_text: str = "",
) -> Dict[str, Any]:
    return {
        "key": key,
        "name": name,
        "type": "number",
        "default": float(default),
        "min": float(minimum),
        "max": float(maximum) if maximum is not None else None,
        "step": float(step),
        "unit": unit,
        "help": help_text,
    }


def _integer_parameter(
    key: str,
    name: str,
    default: int,
    minimum: int = 1,
    maximum: Optional[int] = None,
    unit: str = "次",
    help_text: str = "",
) -> Dict[str, Any]:
    return {
        "key": key,
        "name": name,
        "type": "integer",
        "default": int(default),
        "min": int(minimum),
        "max": int(maximum) if maximum is not None else None,
        "step": 1,
        "unit": unit,
        "help": help_text,
    }


def algorithm_parameter_definitions(algorithm_code: str) -> List[Dict[str, Any]]:
    code = str(algorithm_code or "").strip()
    definition = OBJECT_ALGORITHM_DEFINITIONS.get(code)
    confidence = float(definition.confidence_threshold) if definition is not None else 0.25
    if code == DEFAULT_ALGORITHM_CODE:
        return [
            _number_parameter(
                "person_confidence_threshold",
                "人员置信度",
                MACHINERY_PERSON_CONFIDENCE_THRESHOLD,
                maximum=1.0,
            ),
            _number_parameter(
                "confidence_threshold",
                "机械置信度",
                MACHINERY_CONFIDENCE_THRESHOLD,
                maximum=1.0,
            ),
            _number_parameter(
                "overlap_ratio_threshold",
                "危险区重叠阈值",
                OVERLAP_RATIO_THRESHOLD,
                maximum=1.0,
                step=0.001,
                help_text="人员下半身与机械危险区的最小重叠比例。",
            ),
            _integer_parameter(
                "risk_consecutive_frames",
                "风险确认次数",
                RISK_CONSECUTIVE_FRAMES,
                maximum=30,
            ),
        ]
    if code == "safety_helmet":
        return [
            _number_parameter("person_confidence_threshold", "人员检测阈值", confidence, maximum=1.0),
            _number_parameter("bare_head_confidence_threshold", "裸头违规阈值", confidence, maximum=1.0),
            _number_parameter(
                "helmet_confidence_threshold",
                "安全帽有效证据阈值",
                confidence,
                maximum=1.0,
                help_text="高于该值的安全帽才作为已佩戴证据；调高可能增加未佩戴误报。",
            ),
            _integer_parameter("violation_consecutive_hits", "连续未佩戴确认次数", 1, maximum=30),
        ]
    if code == "cigarette_detection":
        return [
            _number_parameter("confidence_threshold", "香烟置信度阈值", confidence, maximum=1.0),
            _number_parameter("person_confidence_threshold", "人员置信度阈值", confidence, maximum=1.0),
            _integer_parameter("alarm_window_size", "最近检测次数", definition.alarm_window_size, maximum=30),
            _integer_parameter("alarm_min_hits", "最少命中次数", definition.alarm_min_hits, maximum=30),
        ]
    if code == "cellphone_usage":
        return [
            _number_parameter("confidence_threshold", "手机置信度阈值", confidence, maximum=1.0),
            _number_parameter("person_confidence_threshold", "人员置信度阈值", confidence, maximum=1.0),
            _integer_parameter(
                "alarm_min_hits",
                f"确认命中次数（最近 {definition.alarm_window_size} 次）",
                definition.alarm_min_hits,
                maximum=definition.alarm_window_size,
            ),
        ]
    if code in {"fire_detection", "rebar_detection"}:
        target_name = "明火" if code == "fire_detection" else "钢筋"
        hits_name = "连续命中次数" if code == "fire_detection" else "确认命中次数（最近 3 次）"
        return [
            _number_parameter("confidence_threshold", f"{target_name}置信度阈值", confidence, maximum=1.0),
            _number_parameter(
                "minimum_target_area_ratio",
                "最小目标面积占比",
                0.0,
                maximum=1.0,
                step=0.0001,
            ),
            _integer_parameter("alarm_min_hits", hits_name, definition.alarm_min_hits, maximum=definition.alarm_window_size if code == "rebar_detection" else 30),
        ]
    if code in {"smoke_detection", "hanging_basket", "fainting_detection"}:
        target_names = {
            "smoke_detection": "烟雾",
            "hanging_basket": "吊篮",
            "fainting_detection": "晕倒",
        }
        target_name = target_names[code]
        return [
            _number_parameter("confidence_threshold", f"{target_name}置信度阈值", confidence, maximum=1.0),
            _number_parameter(
                "minimum_target_area_ratio",
                "最小目标面积占比",
                0.0,
                maximum=1.0,
                step=0.0001,
            ),
            _number_parameter(
                "alarm_min_duration_seconds",
                "最短持续时间",
                0.0,
                step=0.5,
                unit="秒",
            ),
        ]
    return []


class TaskValidationError(ValueError):
    """Raised when a video-task payload is invalid."""


class TaskNotFoundError(LookupError):
    """Raised when a requested task does not exist."""


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise TaskValidationError(f"{field_name}必须是布尔值")


def normalize_mapping(value: Any) -> Dict[str, str]:
    mapping = value if isinstance(value, dict) else {}
    return {key: str(mapping.get(key, "") or "").strip()[:100] for key in REPORT_MAPPING_KEYS}


def normalize_time_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    try:
        return datetime.strptime(text, "%H:%M").strftime("%H:%M")
    except ValueError as exc:
        raise TaskValidationError(f"{field_name}必须使用 HH:MM 格式") from exc


def normalize_runtime_policy(value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    merged = dict(DEFAULT_RUNTIME_POLICY)
    merged.update(raw)
    try:
        wakeup_seconds = int(merged["event_wakeup_seconds"])
        area_ratio = float(merged["small_target_area_ratio"])
        confirmation_timeout = int(merged["confirmation_timeout_seconds"])
    except (TypeError, ValueError) as exc:
        raise TaskValidationError("节流策略参数格式错误") from exc
    if not 10 <= wakeup_seconds <= 86400:
        raise TaskValidationError("事件唤醒持续时间必须在 10 到 86400 秒之间")
    if not 0.0001 <= area_ratio <= 0.25:
        raise TaskValidationError("小目标面积占比必须在 0.0001 到 0.25 之间")
    if not 1 <= confirmation_timeout <= 60:
        raise TaskValidationError("主码流确认超时必须在 1 到 60 秒之间")
    return {
        "schedule_enabled": parse_bool(merged["schedule_enabled"], "检测时段开关"),
        "schedule_start": normalize_time_text(merged["schedule_start"], "检测开始时间"),
        "schedule_end": normalize_time_text(merged["schedule_end"], "检测结束时间"),
        "event_wakeup_enabled": parse_bool(merged["event_wakeup_enabled"], "事件唤醒开关"),
        "event_wakeup_seconds": wakeup_seconds,
        "main_stream_confirmation_enabled": parse_bool(
            merged["main_stream_confirmation_enabled"], "主码流二次确认开关"
        ),
        "small_target_area_ratio": area_ratio,
        "confirmation_timeout_seconds": confirmation_timeout,
    }


def normalize_algorithm_codes(values: Dict[str, Any]) -> List[str]:
    raw_codes = values.get("algorithm_codes")
    if raw_codes is None:
        enabled = parse_bool(values.get("algorithm_enabled", True), "算法启用状态")
        raw_codes = [DEFAULT_ALGORITHM_CODE] if enabled else []
    if not isinstance(raw_codes, (list, tuple)):
        raise TaskValidationError("algorithm_codes 必须是算法编号数组")
    codes: List[str] = []
    for value in raw_codes:
        code = str(value or "").strip()
        if not code or code in codes:
            continue
        try:
            algorithm_registry.descriptor(code)
        except KeyError as exc:
            raise TaskValidationError(f"算法不存在：{code}") from exc
        codes.append(code)
    return codes


def default_alarm_interval_seconds(algorithm_code: str) -> float:
    code = str(algorithm_code or "").strip()
    if code == DEFAULT_ALGORITHM_CODE:
        return float(ALARM_SAVE_COOLDOWN_SECONDS)
    definition = OBJECT_ALGORITHM_DEFINITIONS.get(code)
    if definition is None:
        raise KeyError(f"未注册的算法：{code}")
    return float(definition.alarm_cooldown_seconds)


def normalize_algorithm_alarm_intervals(
    value: Any,
    algorithm_codes: List[str],
) -> Dict[str, float]:
    raw = {} if value is None else value
    if not isinstance(raw, dict):
        raise TaskValidationError("algorithm_alarm_intervals 必须是算法编号到告警间隔的映射")
    intervals: Dict[str, float] = {}
    for code in algorithm_codes:
        candidate = raw[code] if code in raw else default_alarm_interval_seconds(code)
        if isinstance(candidate, bool):
            raise TaskValidationError(f"{code} 的告警间隔必须是数字")
        try:
            interval = float(candidate)
        except (TypeError, ValueError) as exc:
            raise TaskValidationError(f"{code} 的告警间隔必须是数字") from exc
        if not math.isfinite(interval) or interval < 0:
            raise TaskValidationError(f"{code} 的告警间隔不能小于 0 秒")
        intervals[code] = interval
    return intervals


def normalize_algorithm_parameters(
    value: Any,
    algorithm_codes: List[str],
) -> Dict[str, Dict[str, Any]]:
    raw = {} if value is None else value
    if not isinstance(raw, dict):
        raise TaskValidationError("algorithm_parameters 必须是算法编号到参数配置的映射")
    normalized: Dict[str, Dict[str, Any]] = {}
    for code in algorithm_codes:
        definitions = algorithm_parameter_definitions(code)
        if not definitions:
            continue
        raw_values = raw.get(code, {})
        if not isinstance(raw_values, dict):
            raise TaskValidationError(f"{code} 的算法参数必须是对象")
        parameters: Dict[str, Any] = {}
        for definition in definitions:
            key = definition["key"]
            candidate = raw_values.get(key, definition["default"])
            if isinstance(candidate, bool):
                raise TaskValidationError(f"{definition['name']}必须是数字")
            try:
                number = float(candidate)
            except (TypeError, ValueError) as exc:
                raise TaskValidationError(f"{definition['name']}必须是数字") from exc
            if not math.isfinite(number):
                raise TaskValidationError(f"{definition['name']}必须是有限数字")
            if definition["type"] == "integer":
                if not number.is_integer():
                    raise TaskValidationError(f"{definition['name']}必须是整数")
                parsed: Any = int(number)
            else:
                parsed = number
            minimum = definition.get("min")
            maximum = definition.get("max")
            if minimum is not None and parsed < minimum:
                raise TaskValidationError(f"{definition['name']}不能小于 {minimum:g}")
            if maximum is not None and parsed > maximum:
                raise TaskValidationError(f"{definition['name']}不能大于 {maximum:g}")
            parameters[key] = parsed
        if code == "cigarette_detection" and parameters["alarm_min_hits"] > parameters["alarm_window_size"]:
            raise TaskValidationError("香烟检测的最少命中次数不能大于最近检测次数")
        normalized[code] = parameters
    return normalized


def validate_values(values: Dict[str, Any]) -> Dict[str, Any]:
    task_code = str(values.get("task_code", "") or "").strip()
    if not task_code:
        raise TaskValidationError("任务编号不能为空")
    if len(task_code) > 100:
        raise TaskValidationError("任务编号不能超过 100 个字符")

    try:
        channel_id = int(values.get("channel_id"))
    except (TypeError, ValueError) as exc:
        raise TaskValidationError("请选择视频源") from exc
    if channel_id <= 0:
        raise TaskValidationError("请选择视频源")

    report_url = str(values.get("report_url", "") or "").strip()
    if report_url:
        parts = urlsplit(report_url)
        if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
            raise TaskValidationError("上报地址必须是有效的 HTTP 或 HTTPS 地址")
    if len(report_url) > 1000:
        raise TaskValidationError("上报地址不能超过 1000 个字符")

    algorithm_codes = normalize_algorithm_codes(values)
    return {
        "task_code": task_code,
        "channel_id": channel_id,
        "report_url": report_url,
        "algorithm_enabled": bool(algorithm_codes),
        "algorithm_codes": algorithm_codes,
        "algorithm_alarm_intervals": normalize_algorithm_alarm_intervals(
            values.get("algorithm_alarm_intervals"), algorithm_codes
        ),
        "algorithm_parameters": normalize_algorithm_parameters(
            values.get("algorithm_parameters"), algorithm_codes
        ),
        "report_mapping": normalize_mapping(values.get("report_mapping", {})),
        "runtime_policy": normalize_runtime_policy(values.get("runtime_policy", {})),
    }


class VideoTaskStore:
    """SQLite-backed task configuration and desired-runtime state store."""

    def __init__(self, database_path: Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

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
                CREATE TABLE IF NOT EXISTS video_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_code TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    channel_id INTEGER,
                    report_url TEXT NOT NULL,
                    report_mapping_json TEXT NOT NULL DEFAULT '{}',
                    algorithm_enabled INTEGER NOT NULL DEFAULT 1,
                    algorithm_codes_json TEXT NOT NULL DEFAULT '["machinery_hazard"]',
                    algorithm_alarm_intervals_json TEXT NOT NULL DEFAULT '{}',
                    algorithm_parameters_json TEXT NOT NULL DEFAULT '{}',
                    runtime_policy_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'stopped',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (channel_id) REFERENCES video_channels(id) ON DELETE SET NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(video_tasks)").fetchall()
            }
            if "algorithm_codes_json" not in columns:
                connection.execute(
                    "ALTER TABLE video_tasks ADD COLUMN algorithm_codes_json "
                    "TEXT NOT NULL DEFAULT '[\"machinery_hazard\"]'"
                )
                connection.execute(
                    "UPDATE video_tasks SET algorithm_codes_json = '[]' WHERE algorithm_enabled = 0"
                )
            if "runtime_policy_json" not in columns:
                connection.execute(
                    "ALTER TABLE video_tasks ADD COLUMN runtime_policy_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "algorithm_alarm_intervals_json" not in columns:
                connection.execute(
                    "ALTER TABLE video_tasks ADD COLUMN algorithm_alarm_intervals_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                )
            if "algorithm_parameters_json" not in columns:
                connection.execute(
                    "ALTER TABLE video_tasks ADD COLUMN algorithm_parameters_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                )

    @staticmethod
    def _to_public(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        try:
            mapping = normalize_mapping(json.loads(data.pop("report_mapping_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            mapping = normalize_mapping({})
        try:
            raw_codes = json.loads(data.pop("algorithm_codes_json") or "[]")
            algorithm_codes = [str(value) for value in raw_codes if str(value or "").strip()]
        except (TypeError, ValueError, json.JSONDecodeError):
            algorithm_codes = [DEFAULT_ALGORITHM_CODE] if bool(data.get("algorithm_enabled")) else []
        algorithms = []
        for code in algorithm_codes:
            try:
                descriptor = algorithm_registry.descriptor(code)
            except KeyError:
                continue
            algorithms.append(descriptor.to_dict())
        algorithm_codes = [item["code"] for item in algorithms]
        try:
            runtime_policy = normalize_runtime_policy(json.loads(data.pop("runtime_policy_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError, TaskValidationError):
            runtime_policy = dict(DEFAULT_RUNTIME_POLICY)
        try:
            algorithm_alarm_intervals = normalize_algorithm_alarm_intervals(
                json.loads(data.pop("algorithm_alarm_intervals_json") or "{}"),
                algorithm_codes,
            )
        except (TypeError, ValueError, json.JSONDecodeError, TaskValidationError):
            algorithm_alarm_intervals = normalize_algorithm_alarm_intervals({}, algorithm_codes)
        try:
            algorithm_parameters = normalize_algorithm_parameters(
                json.loads(data.pop("algorithm_parameters_json") or "{}"),
                algorithm_codes,
            )
        except (TypeError, ValueError, json.JSONDecodeError, TaskValidationError):
            algorithm_parameters = normalize_algorithm_parameters({}, algorithm_codes)
        channel_id = data.get("channel_id")
        return {
            "id": int(data["id"]),
            "task_code": data["task_code"],
            "channel_id": int(channel_id) if channel_id is not None else None,
            "channel_name": data.get("channel_name") or "通道已删除",
            "channel_source_type": data.get("channel_source_type") or "",
            "channel_enabled": bool(data.get("channel_enabled")) if channel_id is not None else False,
            "report_url": data["report_url"],
            "report_mapping": mapping,
            "algorithm_enabled": bool(algorithms),
            "algorithm_codes": [item["code"] for item in algorithms],
            "algorithm_alarm_intervals": {
                item["code"]: algorithm_alarm_intervals[item["code"]]
                for item in algorithms
            },
            "algorithm_parameters": algorithm_parameters,
            "algorithms": algorithms,
            "runtime_policy": runtime_policy,
            "status": data["status"] if data["status"] in TASK_STATUSES else TASK_STATUS_STOPPED,
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
        }

    @staticmethod
    def _select_sql(where: str = "") -> str:
        return f"""
            SELECT t.*, c.name AS channel_name, c.source_type AS channel_source_type,
                   c.enabled AS channel_enabled
            FROM video_tasks AS t
            LEFT JOIN video_channels AS c ON c.id = t.channel_id
            {where}
        """

    def _ensure_channel(self, connection: sqlite3.Connection, channel_id: int) -> None:
        row = connection.execute("SELECT id FROM video_channels WHERE id = ?", (int(channel_id),)).fetchone()
        if row is None:
            raise TaskValidationError("所选视频通道不存在")

    def list_tasks(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(self._select_sql("ORDER BY t.id DESC")).fetchall()
        return [self._to_public(row) for row in rows]

    def get_task(self, task_id: int) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(self._select_sql("WHERE t.id = ?"), (int(task_id),)).fetchone()
        if row is None:
            raise TaskNotFoundError("视频任务不存在")
        return self._to_public(row)

    def create_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        values = validate_values(payload)
        timestamp = now_text()
        try:
            with self._lock, self._connect() as connection:
                self._ensure_channel(connection, values["channel_id"])
                cursor = connection.execute(
                    """
                    INSERT INTO video_tasks (
                        task_code, channel_id, report_url, report_mapping_json,
                        algorithm_enabled, algorithm_codes_json, runtime_policy_json,
                        algorithm_alarm_intervals_json, algorithm_parameters_json,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'stopped', ?, ?)
                    """,
                    (
                        values["task_code"],
                        values["channel_id"],
                        values["report_url"],
                        json.dumps(values["report_mapping"], ensure_ascii=False),
                        int(values["algorithm_enabled"]),
                        json.dumps(values["algorithm_codes"], ensure_ascii=False),
                        json.dumps(values["runtime_policy"], ensure_ascii=False),
                        json.dumps(values["algorithm_alarm_intervals"], ensure_ascii=False),
                        json.dumps(values["algorithm_parameters"], ensure_ascii=False),
                        timestamp,
                        timestamp,
                    ),
                )
                task_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise TaskValidationError("任务编号已经存在") from exc
        return self.get_task(task_id)

    def update_task(self, task_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_task(task_id)
        merged = {
            "task_code": current["task_code"],
            "channel_id": current["channel_id"],
            "report_url": current["report_url"],
            "algorithm_enabled": current["algorithm_enabled"],
            "algorithm_codes": current["algorithm_codes"],
            "algorithm_alarm_intervals": current["algorithm_alarm_intervals"],
            "algorithm_parameters": current["algorithm_parameters"],
            "report_mapping": current["report_mapping"],
            "runtime_policy": current["runtime_policy"],
        }
        merged.update(payload)
        if "algorithm_enabled" in payload and "algorithm_codes" not in payload:
            merged["algorithm_codes"] = (
                [DEFAULT_ALGORITHM_CODE]
                if parse_bool(payload["algorithm_enabled"], "算法启用状态")
                else []
            )
        values = validate_values(merged)
        try:
            with self._lock, self._connect() as connection:
                self._ensure_channel(connection, values["channel_id"])
                cursor = connection.execute(
                    """
                    UPDATE video_tasks
                    SET task_code = ?, channel_id = ?, report_url = ?, report_mapping_json = ?,
                        algorithm_enabled = ?, algorithm_codes_json = ?, runtime_policy_json = ?,
                        algorithm_alarm_intervals_json = ?,
                        algorithm_parameters_json = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        values["task_code"],
                        values["channel_id"],
                        values["report_url"],
                        json.dumps(values["report_mapping"], ensure_ascii=False),
                        int(values["algorithm_enabled"]),
                        json.dumps(values["algorithm_codes"], ensure_ascii=False),
                        json.dumps(values["runtime_policy"], ensure_ascii=False),
                        json.dumps(values["algorithm_alarm_intervals"], ensure_ascii=False),
                        json.dumps(values["algorithm_parameters"], ensure_ascii=False),
                        now_text(),
                        int(task_id),
                    ),
                )
                if cursor.rowcount == 0:
                    raise TaskNotFoundError("视频任务不存在")
        except sqlite3.IntegrityError as exc:
            raise TaskValidationError("任务编号已经存在") from exc
        return self.get_task(task_id)

    def set_status(self, task_id: int, status: str) -> Dict[str, Any]:
        normalized = str(status or "").strip().lower()
        if normalized not in TASK_STATUSES:
            raise TaskValidationError("任务状态无效")
        current = self.get_task(task_id)
        if normalized == TASK_STATUS_RUNNING:
            if current["channel_id"] is None:
                raise TaskValidationError("视频通道已删除，请先重新选择视频源")
            if not current["channel_enabled"]:
                raise TaskValidationError("所选视频通道已停用，不能启动任务")
            if not current["algorithm_codes"]:
                raise TaskValidationError("请至少启用一个算法后再启动任务")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE video_tasks SET status = ?, updated_at = ? WHERE id = ?",
                (normalized, now_text(), int(task_id)),
            )
            if cursor.rowcount == 0:
                raise TaskNotFoundError("视频任务不存在")
        return self.get_task(task_id)

    def delete_task(self, task_id: int) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM video_tasks WHERE id = ?", (int(task_id),))
            if cursor.rowcount == 0:
                raise TaskNotFoundError("视频任务不存在")

    def channel_delete_block_reason(self, channel_id: int) -> Optional[str]:
        """Require users to remove every referencing task before a channel."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS running
                FROM video_tasks WHERE channel_id = ?
                """,
                (int(channel_id),),
            ).fetchone()
        total = int(row["total"] or 0)
        running = int(row["running"] or 0)
        if running:
            return f"该视频通道仍被 {running} 个已启动任务使用，请先停止并删除对应任务"
        if total:
            return f"该视频通道仍关联 {total} 个视频任务，请先删除对应任务"
        return None


def register_video_task_routes(app: Any, store: VideoTaskStore, runtime: Any = None) -> None:
    """Attach task page and JSON APIs to a FastAPI application."""
    from fastapi import HTTPException, Request
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def handle_store_error(exc: BaseException) -> None:
        if isinstance(exc, TaskNotFoundError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, TaskValidationError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise exc

    @app.get("/tasks", response_class=HTMLResponse, include_in_schema=False)
    def tasks_page(request: Request) -> Any:
        return templates.TemplateResponse(request=request, name="tasks.html", context={"page_title": "视频任务"})

    @app.get("/api/video-tasks/meta")
    def video_task_meta() -> Dict[str, Any]:
        return {
            "default_report_url": DEFAULT_REPORT_URL,
            "algorithms": [
                {
                    "code": item["code"],
                    "name": item["name"],
                    "description": item.get("description", ""),
                    "default_alarm_interval_seconds": default_alarm_interval_seconds(item["code"]),
                    "parameters": algorithm_parameter_definitions(item["code"]),
                }
                for item in algorithm_registry.list_public()
            ],
            "report_mapping_fields": list(REPORT_MAPPING_KEYS),
            "report_payload_rules": {
                "task_code": "视频任务编号仅用于区分任务，不参与告警上报",
                "camera_id": "自动读取视频通道 CameraID",
                "event_type": "自动使用触发告警的算法名称",
                "event_time": "自动使用告警产生时间",
                "image": "自动将带框告警图片转换为 Base64",
            },
            "worker_connected": runtime is not None,
            "camera_event_wakeup_endpoint": "/api/camera-events/wake",
        }

    @app.post("/api/camera-events/wake")
    def camera_event_wakeup(payload: Dict[str, Any]) -> Dict[str, Any]:
        if runtime is None:
            raise HTTPException(status_code=503, detail="任务执行器未连接")
        try:
            return runtime.wake_from_camera_event(payload)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/video-tasks")
    def list_video_tasks() -> Dict[str, Any]:
        tasks = runtime.list_tasks() if runtime is not None else store.list_tasks()
        return {"items": tasks, "total": len(tasks)}

    @app.post("/api/video-tasks", status_code=201)
    def create_video_task(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return store.create_task(payload)
        except (TaskNotFoundError, TaskValidationError) as exc:
            handle_store_error(exc)
            raise AssertionError("unreachable")

    @app.put("/api/video-tasks/{task_id}")
    def update_video_task(task_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if runtime is not None and runtime.is_active(task_id):
                raise TaskValidationError("请先停止任务后再编辑配置")
            return store.update_task(task_id, payload)
        except (TaskNotFoundError, TaskValidationError) as exc:
            handle_store_error(exc)
            raise AssertionError("unreachable")

    @app.patch("/api/video-tasks/{task_id}/status")
    def set_video_task_status(task_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "status" not in payload:
            raise HTTPException(status_code=400, detail="请求体需要 status 字段")
        try:
            status = str(payload["status"]).strip().lower()
            if runtime is None:
                return store.set_status(task_id, status)
            if status == TASK_STATUS_RUNNING:
                return runtime.start_task(task_id)
            if status == TASK_STATUS_STOPPED:
                return runtime.stop_task(task_id)
            raise TaskValidationError("任务状态无效")
        except (TaskNotFoundError, TaskValidationError) as exc:
            handle_store_error(exc)
            raise AssertionError("unreachable")

    @app.delete("/api/video-tasks/{task_id}")
    def delete_video_task(task_id: int) -> Dict[str, bool]:
        try:
            if runtime is not None:
                runtime.stop_task(task_id)
            store.delete_task(task_id)
        except (TaskNotFoundError, TaskValidationError) as exc:
            handle_store_error(exc)
        return {"success": True}
