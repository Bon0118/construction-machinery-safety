"""Local video-alarm index, retention, and Web routes."""

import json
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import config
import push_alarm
from alarm_event_schema import ALARM_IMAGE_FILENAME, EVENT_DIR_PREFIX, METADATA_FILENAME
from video_channels import DEFAULT_DATABASE_PATH, TEMPLATES_DIR


ALARM_TYPE_MACHINERY_HAZARD = "机械伤害预警"
ALARM_CONTENT_MACHINERY_HAZARD = "检测到人员进入工程机械危险区域"
REPORT_STATUS_LOCAL_ONLY = "local_only"
REPORT_STATUS_PENDING = "pending"
REPORT_STATUS_REPORTING = "reporting"
REPORT_STATUS_SUCCESS = "success"
REPORT_STATUS_FAILED = "failed"
REPORT_STATUS_REVIEW_BLOCKED = "review_blocked"
REPORT_STATUSES = {
    REPORT_STATUS_LOCAL_ONLY,
    REPORT_STATUS_PENDING,
    REPORT_STATUS_REPORTING,
    REPORT_STATUS_SUCCESS,
    REPORT_STATUS_FAILED,
    REPORT_STATUS_REVIEW_BLOCKED,
}
REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_REVIEWING = "reviewing"
REVIEW_STATUS_CONFIRMED = "confirmed"
REVIEW_STATUS_REJECTED = "rejected"
REVIEW_STATUS_UNCERTAIN = "uncertain"
REVIEW_STATUSES = {
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REVIEWING,
    REVIEW_STATUS_CONFIRMED,
    REVIEW_STATUS_REJECTED,
    REVIEW_STATUS_UNCERTAIN,
}
REVIEW_RESULTS = {
    REVIEW_STATUS_CONFIRMED,
    REVIEW_STATUS_REJECTED,
    REVIEW_STATUS_UNCERTAIN,
}


class AlarmValidationError(ValueError):
    """Raised when an alarm query or mutation is invalid."""


class AlarmNotFoundError(LookupError):
    """Raised when an alarm record does not exist."""


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class VideoAlarmStore:
    """Index the detector's event directories in SQLite for the Web UI."""

    def __init__(
        self,
        database_path: Path = DEFAULT_DATABASE_PATH,
        output_dir: Path = config.ALARM_OUTPUT_DIR,
        max_records: int = 1000,
    ) -> None:
        self.database_path = Path(database_path)
        self.output_dir = Path(output_dir)
        self.max_records = max(1, int(max_records))
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS video_alarms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    task_id INTEGER,
                    task_code TEXT NOT NULL DEFAULT '',
                    channel_id INTEGER,
                    channel_name TEXT NOT NULL DEFAULT '',
                    alarm_type TEXT NOT NULL,
                    alarm_time TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    metadata_path TEXT NOT NULL DEFAULT '',
                    report_url TEXT NOT NULL DEFAULT '',
                    report_status TEXT NOT NULL DEFAULT 'local_only',
                    report_error TEXT NOT NULL DEFAULT '',
                    alarm_content TEXT NOT NULL DEFAULT '',
                    review_status TEXT NOT NULL DEFAULT 'uncertain',
                    review_probability REAL,
                    review_confidence REAL,
                    review_evidence_quality TEXT NOT NULL DEFAULT '',
                    review_summary TEXT NOT NULL DEFAULT '',
                    review_reason_codes TEXT NOT NULL DEFAULT '[]',
                    review_error TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT,
                    review_model TEXT NOT NULL DEFAULT '',
                    review_prompt_version TEXT NOT NULL DEFAULT '',
                    review_attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES video_tasks(id) ON DELETE SET NULL,
                    FOREIGN KEY (channel_id) REFERENCES video_channels(id) ON DELETE SET NULL
                )
                """
            )
            existing_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(video_alarms)").fetchall()
            }
            migrations = {
                "review_status": "TEXT NOT NULL DEFAULT 'uncertain'",
                "review_probability": "REAL",
                "review_confidence": "REAL",
                "review_evidence_quality": "TEXT NOT NULL DEFAULT ''",
                "review_summary": "TEXT NOT NULL DEFAULT ''",
                "review_reason_codes": "TEXT NOT NULL DEFAULT '[]'",
                "review_error": "TEXT NOT NULL DEFAULT ''",
                "reviewed_at": "TEXT",
                "review_model": "TEXT NOT NULL DEFAULT ''",
                "review_prompt_version": "TEXT NOT NULL DEFAULT ''",
                "review_attempts": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, definition in migrations.items():
                if column not in existing_columns:
                    connection.execute(f"ALTER TABLE video_alarms ADD COLUMN {column} {definition}")
            connection.execute(
                "UPDATE video_alarms SET report_status = 'pending' WHERE report_status = 'reporting'"
            )
            connection.execute(
                "UPDATE video_alarms SET review_status = 'pending' WHERE review_status = 'reviewing'"
            )
            connection.execute(
                """
                UPDATE video_alarms
                SET review_summary = '平台复核功能上线前的历史告警，未经过大模型复核。'
                WHERE review_status = 'uncertain' AND TRIM(review_summary) = ''
                """
            )
            connection.execute(
                """
                UPDATE video_alarms SET report_status = 'review_blocked'
                WHERE report_status = 'pending' AND review_status <> 'confirmed'
                  AND TRIM(report_url) <> ''
                """
            )

    @staticmethod
    def _row_to_public(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        try:
            reason_codes = json.loads(data.get("review_reason_codes") or "[]")
            if not isinstance(reason_codes, list):
                reason_codes = []
        except (TypeError, ValueError, json.JSONDecodeError):
            reason_codes = []
        review_status = data.get("review_status") or REVIEW_STATUS_UNCERTAIN
        report_status = data.get("report_status") or REPORT_STATUS_LOCAL_ONLY
        report_url = data.get("report_url") or ""
        return {
            "id": int(data["id"]),
            "event_id": data["event_id"],
            "task_id": int(data["task_id"]) if data.get("task_id") is not None else None,
            "task_code": data.get("task_code") or "未关联任务",
            "channel_id": int(data["channel_id"]) if data.get("channel_id") is not None else None,
            "channel_name": data.get("channel_name") or "未关联通道",
            "alarm_type": data["alarm_type"],
            "alarm_time": data["alarm_time"],
            "report_url": report_url,
            "report_status": report_status,
            "report_error": data.get("report_error") or "",
            "alarm_content": data.get("alarm_content") or ALARM_CONTENT_MACHINERY_HAZARD,
            "review_status": review_status,
            "review_probability": data.get("review_probability"),
            "review_confidence": data.get("review_confidence"),
            "review_evidence_quality": data.get("review_evidence_quality") or "",
            "review_summary": data.get("review_summary") or "",
            "review_reason_codes": reason_codes,
            "review_error": data.get("review_error") or "",
            "reviewed_at": data.get("reviewed_at"),
            "review_model": data.get("review_model") or "",
            "review_prompt_version": data.get("review_prompt_version") or "",
            "can_manual_report": bool(
                review_status == REVIEW_STATUS_UNCERTAIN
                and str(report_url).strip()
                and report_status not in {REPORT_STATUS_SUCCESS, REPORT_STATUS_REPORTING}
            ),
            "image_url": f"/api/video-alarms/{int(data['id'])}/image",
            "created_at": data["created_at"],
        }

    def _event_dirs(self) -> List[Path]:
        prefix = f"{EVENT_DIR_PREFIX}_"
        try:
            return sorted(
                (path for path in self.output_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)),
                key=lambda path: (path.stat().st_mtime, path.name),
            )
        except FileNotFoundError:
            return []

    @staticmethod
    def _load_metadata(event_dir: Path) -> Tuple[Dict[str, Any], Optional[Path]]:
        metadata_path = event_dir / METADATA_FILENAME
        if not metadata_path.is_file():
            return {}, None
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            return (data if isinstance(data, dict) else {}), metadata_path
        except (OSError, ValueError, json.JSONDecodeError):
            return {}, metadata_path

    @staticmethod
    def _alarm_image(event_dir: Path, metadata: Dict[str, Any]) -> Optional[Path]:
        # The gallery must always show the annotated evidence image.  Falling
        # back to original.jpg would hide the recognition boxes and make an
        # incomplete event look valid.
        path = event_dir / ALARM_IMAGE_FILENAME
        return path if path.is_file() else None

    @staticmethod
    def _normalize_alarm_type(value: Any) -> str:
        text = str(value or "").strip()
        if text in {"", "机械伤害风险", "注意机械伤害风险"}:
            return ALARM_TYPE_MACHINERY_HAZARD
        return text

    @staticmethod
    def _channel_matches(row: sqlite3.Row, source: str, camera: str) -> bool:
        if camera and camera.casefold() == str(row["name"] or "").casefold():
            return True
        source_type = str(row["source_type"] or "")
        if source_type == "camera_index_code":
            code = str(row["camera_index_code"] or "")
            return bool(code and source in {code, f"cameraIndexCode:{code}"})
        return bool(source and source == str(row["source_url"] or ""))

    def _resolve_relations(self, connection: sqlite3.Connection, metadata: Dict[str, Any]) -> Dict[str, Any]:
        task_row: Optional[sqlite3.Row] = None
        channel_row: Optional[sqlite3.Row] = None
        raw_task_id = metadata.get("task_id")
        raw_task_code = str(metadata.get("task_code", "") or "").strip()
        raw_channel_id = metadata.get("channel_id")

        if raw_task_id is not None:
            try:
                task_row = connection.execute("SELECT * FROM video_tasks WHERE id = ?", (int(raw_task_id),)).fetchone()
            except (TypeError, ValueError):
                task_row = None
        if task_row is None and raw_task_code:
            task_row = connection.execute(
                "SELECT * FROM video_tasks WHERE task_code = ? COLLATE NOCASE",
                (raw_task_code,),
            ).fetchone()

        if raw_channel_id is not None:
            try:
                channel_row = connection.execute("SELECT * FROM video_channels WHERE id = ?", (int(raw_channel_id),)).fetchone()
            except (TypeError, ValueError):
                channel_row = None
        if channel_row is None and task_row is not None and task_row["channel_id"] is not None:
            channel_row = connection.execute(
                "SELECT * FROM video_channels WHERE id = ?",
                (int(task_row["channel_id"]),),
            ).fetchone()

        if channel_row is None:
            source = str(metadata.get("source", "") or "").strip()
            camera = str(metadata.get("camera", "") or "").strip()
            rows = connection.execute("SELECT * FROM video_channels ORDER BY id").fetchall()
            channel_row = next((row for row in rows if self._channel_matches(row, source, camera)), None)

        if task_row is None and channel_row is not None:
            task_row = connection.execute(
                """
                SELECT * FROM video_tasks WHERE channel_id = ?
                ORDER BY CASE WHEN status = 'running' THEN 0 ELSE 1 END, id DESC LIMIT 1
                """,
                (int(channel_row["id"]),),
            ).fetchone()

        return {
            "task_id": int(task_row["id"]) if task_row is not None else None,
            "task_code": str(task_row["task_code"]) if task_row is not None else raw_task_code,
            "channel_id": int(channel_row["id"]) if channel_row is not None else None,
            "channel_name": (
                str(channel_row["name"])
                if channel_row is not None
                else str(metadata.get("camera", "") or "")
            ),
            "report_url": str(task_row["report_url"] or "") if task_row is not None else "",
        }

    def sync_from_output(self) -> int:
        """Import new detector event directories and remove stale index rows."""
        imported = 0
        with self._lock, self._connect() as connection:
            existing = {
                str(row["event_id"])
                for row in connection.execute("SELECT event_id FROM video_alarms").fetchall()
            }
            for event_dir in self._event_dirs():
                metadata, metadata_path = self._load_metadata(event_dir)
                event_id = str(metadata.get("event_id") or event_dir.name)
                if event_id in existing:
                    continue
                image_path = self._alarm_image(event_dir, metadata)
                if image_path is None:
                    continue
                relations = self._resolve_relations(connection, metadata)
                report_url = str(metadata.get("report_url") or relations["report_url"] or "").strip()
                report_status = str(metadata.get("report_status") or "").strip().lower()
                if report_status not in REPORT_STATUSES:
                    report_status = REPORT_STATUS_PENDING if report_url else REPORT_STATUS_LOCAL_ONLY
                review_status = str(metadata.get("review_status") or REVIEW_STATUS_PENDING).strip().lower()
                if review_status not in REVIEW_STATUSES:
                    review_status = REVIEW_STATUS_PENDING
                if report_url and report_status not in {REPORT_STATUS_SUCCESS, REPORT_STATUS_FAILED}:
                    report_status = (
                        REPORT_STATUS_PENDING
                        if review_status == REVIEW_STATUS_CONFIRMED
                        else REPORT_STATUS_REVIEW_BLOCKED
                    )
                alarm_time = str(metadata.get("time") or "").strip()
                if not alarm_time:
                    alarm_time = datetime.fromtimestamp(event_dir.stat().st_mtime).astimezone().isoformat(timespec="seconds")
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO video_alarms (
                        event_id, task_id, task_code, channel_id, channel_name, alarm_type,
                        alarm_time, image_path, metadata_path, report_url, report_status,
                        report_error, alarm_content, review_status, review_probability,
                        review_confidence, review_evidence_quality, review_summary,
                        review_reason_codes, review_error, reviewed_at, review_model,
                        review_prompt_version, review_attempts, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        relations["task_id"],
                        relations["task_code"],
                        relations["channel_id"],
                        relations["channel_name"],
                        self._normalize_alarm_type(metadata.get("event_type")),
                        alarm_time,
                        str(image_path.resolve()),
                        str(metadata_path.resolve()) if metadata_path else "",
                        report_url,
                        report_status,
                        str(metadata.get("report_error", "") or ""),
                        str(metadata.get("alarm_content") or ALARM_CONTENT_MACHINERY_HAZARD),
                        review_status,
                        metadata.get("review_probability"),
                        metadata.get("review_confidence"),
                        str(metadata.get("review_evidence_quality", "") or ""),
                        str(metadata.get("review_summary", "") or ""),
                        json.dumps(metadata.get("review_reason_codes") or [], ensure_ascii=False),
                        str(metadata.get("review_error", "") or ""),
                        metadata.get("reviewed_at"),
                        str(metadata.get("review_model", "") or ""),
                        str(metadata.get("review_prompt_version", "") or ""),
                        int(metadata.get("review_attempts", 0) or 0),
                        now_text(),
                    ),
                )
                if cursor.rowcount > 0:
                    imported += 1
                existing.add(event_id)

            stale_ids = []
            for row in connection.execute("SELECT id, image_path FROM video_alarms").fetchall():
                indexed_path = Path(str(row["image_path"]))
                alarm_path = indexed_path.parent / ALARM_IMAGE_FILENAME
                if not alarm_path.is_file():
                    stale_ids.append(int(row["id"]))
                    continue
                if indexed_path != alarm_path:
                    connection.execute(
                        "UPDATE video_alarms SET image_path = ? WHERE id = ?",
                        (str(alarm_path.resolve()), int(row["id"])),
                    )
            if stale_ids:
                connection.executemany("DELETE FROM video_alarms WHERE id = ?", ((value,) for value in stale_ids))

        self.enforce_limit()
        return imported

    @staticmethod
    def _parse_report_time(value: str) -> datetime:
        text = str(value or "").strip()
        if not text:
            return datetime.now()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return push_alarm.parse_event_time(text)

    @staticmethod
    def _update_report_metadata(metadata_path: str, updates: Dict[str, Any]) -> None:
        path = Path(str(metadata_path or ""))
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
            data.update(updates)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError, json.JSONDecodeError):
            return

    def pending_review_event_ids(self, limit: int = 1000) -> List[str]:
        """Return persisted review work so a restart can resume safely."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id FROM video_alarms
                WHERE review_status = 'pending'
                ORDER BY alarm_time, id LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [str(row["event_id"]) for row in rows]

    def claim_review(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Atomically claim one alarm for the single live-review worker."""
        normalized = str(event_id or "").strip()
        if not normalized:
            return None
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM video_alarms WHERE event_id = ?",
                (normalized,),
            ).fetchone()
            if row is None or str(row["review_status"]) != REVIEW_STATUS_PENDING:
                return None
            connection.execute(
                """
                UPDATE video_alarms
                SET review_status = 'reviewing', review_error = '',
                    review_attempts = review_attempts + 1
                WHERE event_id = ?
                """,
                (normalized,),
            )
            claimed = dict(row)
            claimed["review_status"] = REVIEW_STATUS_REVIEWING
            claimed["review_attempts"] = int(claimed.get("review_attempts") or 0) + 1
            return claimed

    def complete_review(
        self,
        event_id: str,
        review: Dict[str, Any],
        *,
        model: str,
        prompt_version: str,
        error: str = "",
    ) -> Dict[str, Any]:
        """Persist a final review and release only confirmed alarms for reporting."""
        verdict = str(review.get("verdict") or "").strip().lower()
        if verdict not in REVIEW_RESULTS:
            raise AlarmValidationError(f"无效复核结果：{verdict}")
        reviewed_at = now_text()
        reason_codes = review.get("reason_codes") or []
        if not isinstance(reason_codes, list):
            reason_codes = []
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM video_alarms WHERE event_id = ?",
                (str(event_id),),
            ).fetchone()
            if row is None:
                raise AlarmNotFoundError("视频告警不存在")
            report_status = str(row["report_status"] or "")
            report_url = str(row["report_url"] or "").strip()
            if report_status != REPORT_STATUS_SUCCESS:
                if not report_url:
                    report_status = REPORT_STATUS_LOCAL_ONLY
                elif verdict == REVIEW_STATUS_CONFIRMED:
                    report_status = REPORT_STATUS_PENDING
                else:
                    report_status = REPORT_STATUS_REVIEW_BLOCKED
            connection.execute(
                """
                UPDATE video_alarms SET
                    review_status = ?, review_probability = ?, review_confidence = ?,
                    review_evidence_quality = ?, review_summary = ?,
                    review_reason_codes = ?, review_error = ?, reviewed_at = ?,
                    review_model = ?, review_prompt_version = ?, report_status = ?,
                    report_error = CASE WHEN ? = 'review_blocked' THEN '' ELSE report_error END
                WHERE event_id = ?
                """,
                (
                    verdict,
                    review.get("alarm_probability"),
                    review.get("verdict_confidence"),
                    str(review.get("evidence_quality") or ""),
                    str(review.get("summary") or "")[:2000],
                    json.dumps(reason_codes, ensure_ascii=False),
                    str(error or "")[:2000],
                    reviewed_at,
                    str(model or "")[:200],
                    str(prompt_version or "")[:100],
                    report_status,
                    report_status,
                    str(event_id),
                ),
            )
            metadata_path = str(row["metadata_path"] or "")
            alarm_id = int(row["id"])
        self._update_report_metadata(
            metadata_path,
            {
                "review_status": verdict,
                "review_probability": review.get("alarm_probability"),
                "review_confidence": review.get("verdict_confidence"),
                "review_evidence_quality": str(review.get("evidence_quality") or ""),
                "review_summary": str(review.get("summary") or ""),
                "review_reason_codes": reason_codes,
                "review_error": str(error or ""),
                "reviewed_at": reviewed_at,
                "review_model": str(model or ""),
                "review_prompt_version": str(prompt_version or ""),
                "report_status": report_status,
            },
        )
        return {"id": alarm_id, "event_id": str(event_id), "verdict": verdict, "report_status": report_status}

    def fail_review(self, event_id: str, error: str, *, model: str, prompt_version: str) -> Dict[str, Any]:
        """Convert infrastructure/model failures to a conservative uncertain result."""
        message = str(error or "复核失败")[:2000]
        return self.complete_review(
            event_id,
            {
                "verdict": REVIEW_STATUS_UNCERTAIN,
                "alarm_probability": 0.5,
                "verdict_confidence": 0.0,
                "evidence_quality": "low",
                "summary": f"大模型复核未完成：{message}",
                "reason_codes": ["review_failed", "insufficient_evidence"],
            },
            model=model,
            prompt_version=prompt_version,
            error=message,
        )

    def report_alarm(self, alarm_id: int, *, manual_override: bool = False) -> Dict[str, Any]:
        """Send one pending alarm using its image, channel CameraID and task URL."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT a.*, c.camera_id AS channel_camera_id
                FROM video_alarms AS a
                LEFT JOIN video_channels AS c ON c.id = a.channel_id
                WHERE a.id = ?
                """,
                (int(alarm_id),),
            ).fetchone()
            if row is None:
                raise AlarmNotFoundError("视频告警不存在")
            review_status = str(row["review_status"] or "")
            if manual_override and review_status != REVIEW_STATUS_UNCERTAIN:
                raise AlarmValidationError("只有复核结果为 uncertain 的告警允许手动上报")
            if not manual_override and review_status != REVIEW_STATUS_CONFIRMED:
                return {
                    "id": int(row["id"]),
                    "status": REPORT_STATUS_REVIEW_BLOCKED,
                    "sent": False,
                    "review_status": review_status,
                }
            current_status = str(row["report_status"] or "")
            allowed_statuses = (
                {
                    REPORT_STATUS_REVIEW_BLOCKED,
                    REPORT_STATUS_PENDING,
                    REPORT_STATUS_FAILED,
                    REPORT_STATUS_LOCAL_ONLY,
                }
                if manual_override
                else {REPORT_STATUS_PENDING, REPORT_STATUS_FAILED}
            )
            if current_status not in allowed_statuses:
                return {"id": int(row["id"]), "status": current_status, "sent": False}
            report_url = str(row["report_url"] or "").strip()
            if not report_url:
                if manual_override:
                    raise AlarmValidationError("该告警未配置上报地址，无法手动上报")
                connection.execute(
                    "UPDATE video_alarms SET report_status = ?, report_error = '' WHERE id = ?",
                    (REPORT_STATUS_LOCAL_ONLY, int(alarm_id)),
                )
                return {"id": int(row["id"]), "status": REPORT_STATUS_LOCAL_ONLY, "sent": False}
            connection.execute(
                "UPDATE video_alarms SET report_status = ?, report_error = '' WHERE id = ?",
                (REPORT_STATUS_REPORTING, int(alarm_id)),
            )
            alarm = dict(row)

        status = REPORT_STATUS_SUCCESS
        error = ""
        response_status: Optional[int] = None
        try:
            image_path = Path(str(alarm["image_path"])).resolve()
            if not image_path.is_file():
                raise FileNotFoundError("告警图片不存在")
            camera_id = str(alarm.get("channel_camera_id") or "").strip()
            if not camera_id:
                raise ValueError("视频通道未配置 CameraID，无法上报告警")
            payload = push_alarm.build_payload(
                image_path=image_path,
                camera_id=camera_id,
                event_time=self._parse_report_time(str(alarm["alarm_time"])),
                event_type=str(alarm["alarm_type"] or ALARM_TYPE_MACHINERY_HAZARD),
            )
            result = push_alarm.push_alarm(report_url, payload, push_alarm.TIMEOUT_SECONDS)
            response_status = int(result.get("status", 0) or 0)
            if not bool(result.get("success")):
                raise RuntimeError(f"上报接口返回状态 {response_status or 'unknown'}")
        except Exception as exc:
            status = REPORT_STATUS_FAILED
            error = str(exc)[:1000]

        reported_at = now_text()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE video_alarms SET report_status = ?, report_error = ? WHERE id = ?",
                (status, error, int(alarm_id)),
            )
        self._update_report_metadata(
            str(alarm.get("metadata_path") or ""),
            {
                "report_status": status,
                "report_error": error,
                "reported_at": reported_at,
                "report_http_status": response_status,
                "manual_report": bool(manual_override),
                "manual_report_attempted_at": reported_at if manual_override else None,
            },
        )
        return {
            "id": int(alarm_id),
            "status": status,
            "sent": status == REPORT_STATUS_SUCCESS,
            "error": error,
            "http_status": response_status,
            "manual_override": bool(manual_override),
        }

    def manual_report_alarm(self, alarm_id: int) -> Dict[str, Any]:
        """Explicit operator override for one uncertain alarm."""
        return self.report_alarm(int(alarm_id), manual_override=True)

    def report_pending(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id FROM video_alarms
                WHERE review_status = 'confirmed'
                  AND report_status IN ('pending', 'failed') AND TRIM(report_url) <> ''
                ORDER BY alarm_time, id LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [self.report_alarm(int(row["id"])) for row in rows]

    def list_alarms(
        self,
        task_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        alarm_date: str = "",
        review_status: str = "",
        page: int = 1,
        page_size: int = 12,
    ) -> Dict[str, Any]:
        self.sync_from_output()
        page = max(1, int(page))
        page_size = max(1, min(48, int(page_size)))
        where = []
        values: List[Any] = []
        if task_id is not None:
            where.append("task_id = ?")
            values.append(int(task_id))
        if channel_id is not None:
            where.append("channel_id = ?")
            values.append(int(channel_id))
        date_text = str(alarm_date or "").strip()
        if date_text:
            try:
                datetime.strptime(date_text, "%Y-%m-%d")
            except ValueError as exc:
                raise AlarmValidationError("告警日期格式必须是 YYYY-MM-DD") from exc
            where.append("substr(alarm_time, 1, 10) = ?")
            values.append(date_text)
        review_text = str(review_status or "").strip().lower()
        if review_text:
            if review_text not in REVIEW_RESULTS:
                raise AlarmValidationError("复核结果必须是 confirmed、rejected 或 uncertain")
            where.append("review_status = ?")
            values.append(review_text)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM video_alarms {clause}", values).fetchone()[0])
            pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, pages)
            rows = connection.execute(
                f"SELECT * FROM video_alarms {clause} ORDER BY alarm_time DESC, id DESC LIMIT ? OFFSET ?",
                [*values, page_size, (page - 1) * page_size],
            ).fetchall()
        return {
            "items": [self._row_to_public(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
            "max_records": self.max_records,
        }

    def image_path(self, alarm_id: int) -> Path:
        with self._connect() as connection:
            row = connection.execute("SELECT image_path FROM video_alarms WHERE id = ?", (int(alarm_id),)).fetchone()
        if row is None:
            raise AlarmNotFoundError("视频告警不存在")
        path = Path(str(row["image_path"])).resolve()
        output_root = self.output_dir.resolve()
        if output_root not in path.parents or not path.is_file():
            raise AlarmNotFoundError("告警图片不存在")
        return path

    def _remove_event_dir(self, image_path: str) -> None:
        output_root = self.output_dir.resolve()
        event_dir = Path(image_path).resolve().parent
        prefix = f"{EVENT_DIR_PREFIX}_"
        if event_dir.parent != output_root or not event_dir.name.startswith(prefix):
            return
        if event_dir.exists():
            shutil.rmtree(event_dir)

    def delete_alarms(self, alarm_ids: Iterable[int]) -> int:
        ids = sorted({int(value) for value in alarm_ids if int(value) > 0})
        if not ids:
            raise AlarmValidationError("请选择要删除的告警记录")
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT id, image_path FROM video_alarms WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            for row in rows:
                self._remove_event_dir(str(row["image_path"]))
            connection.execute(f"DELETE FROM video_alarms WHERE id IN ({placeholders})", ids)
        return len(rows)

    def clear_all(self) -> int:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT id, image_path FROM video_alarms").fetchall()
            for row in rows:
                self._remove_event_dir(str(row["image_path"]))
            connection.execute("DELETE FROM video_alarms")
        return len(rows)

    def enforce_limit(self) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM video_alarms ORDER BY alarm_time DESC, id DESC LIMIT -1 OFFSET ?",
                (self.max_records,),
            ).fetchall()
        ids = [int(row["id"]) for row in rows]
        return self.delete_alarms(ids) if ids else 0


def register_video_alarm_routes(app: Any, store: VideoAlarmStore) -> None:
    """Attach alarm gallery page and APIs to FastAPI."""
    from fastapi import HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.templating import Jinja2Templates

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def handle_error(exc: BaseException) -> None:
        if isinstance(exc, AlarmNotFoundError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, AlarmValidationError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise exc

    @app.get("/alarms", response_class=HTMLResponse, include_in_schema=False)
    def alarms_page(request: Request) -> Any:
        return templates.TemplateResponse(request=request, name="alarms.html", context={"page_title": "视频告警"})

    @app.get("/api/video-alarms")
    def list_video_alarms(
        task_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        alarm_date: str = "",
        review_status: str = "",
        page: int = 1,
        page_size: int = 12,
    ) -> Dict[str, Any]:
        try:
            return store.list_alarms(
                task_id=task_id,
                channel_id=channel_id,
                alarm_date=alarm_date,
                review_status=review_status,
                page=page,
                page_size=page_size,
            )
        except (AlarmNotFoundError, AlarmValidationError) as exc:
            handle_error(exc)
            raise AssertionError("unreachable")

    @app.get("/api/video-alarms/{alarm_id}/image", include_in_schema=False)
    def video_alarm_image(alarm_id: int) -> FileResponse:
        try:
            path = store.image_path(alarm_id)
        except (AlarmNotFoundError, AlarmValidationError) as exc:
            handle_error(exc)
            raise AssertionError("unreachable")
        return FileResponse(path=str(path))

    @app.post("/api/video-alarms/{alarm_id}/manual-report")
    def manual_report_video_alarm(alarm_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("confirmed") is not True:
            raise HTTPException(status_code=400, detail="手动上报需要明确二次确认")
        try:
            result = store.manual_report_alarm(alarm_id)
        except (AlarmNotFoundError, AlarmValidationError) as exc:
            handle_error(exc)
            raise AssertionError("unreachable")
        return {"success": bool(result.get("sent")), **result}

    @app.post("/api/video-alarms/delete")
    def delete_video_alarms(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_ids = payload.get("ids", [])
        if not isinstance(raw_ids, list):
            raise HTTPException(status_code=400, detail="ids 必须是数组")
        try:
            deleted = store.delete_alarms(raw_ids)
        except (AlarmNotFoundError, AlarmValidationError, TypeError, ValueError) as exc:
            if isinstance(exc, (TypeError, ValueError)) and not isinstance(exc, AlarmValidationError):
                exc = AlarmValidationError("告警记录编号无效")
            handle_error(exc)
            raise AssertionError("unreachable")
        return {"success": True, "deleted": deleted}

    @app.delete("/api/video-alarms")
    def clear_video_alarms() -> Dict[str, Any]:
        return {"success": True, "deleted": store.clear_all()}
