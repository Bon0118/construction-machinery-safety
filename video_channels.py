"""Video-channel persistence, validation, probing, and Web routes."""

import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import cv2
import numpy as np

import config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "platform.db"
WEB_ROOT = PROJECT_ROOT / "web"
TEMPLATES_DIR = WEB_ROOT / "templates"
STATIC_DIR = WEB_ROOT / "static"
SOURCE_TYPE_CAMERA_INDEX_CODE = "camera_index_code"
SOURCE_TYPE_RTSP = "rtsp"
SOURCE_TYPE_LOCAL_FILE = "local_file"
SOURCE_TYPES = {
    SOURCE_TYPE_CAMERA_INDEX_CODE,
    SOURCE_TYPE_RTSP,
    SOURCE_TYPE_LOCAL_FILE,
}
MASKED_PASSWORD = "******"
CONNECTION_TEST_TIMEOUT_SECONDS = 60.0
STREAM_PROFILE_MAIN = "main"
STREAM_PROFILE_SUB = "sub"
STREAM_PROFILES = {STREAM_PROFILE_MAIN, STREAM_PROFILE_SUB}


class ChannelValidationError(ValueError):
    """Raised when a video-channel payload is invalid."""


class ChannelNotFoundError(LookupError):
    """Raised when a requested video channel does not exist."""


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def mask_stream_url(value: str) -> str:
    """Mask a password embedded in a URL without changing other components."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    if not parts.scheme or parts.password is None:
        return text

    hostname = parts.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    username = quote(unquote(parts.username or ""), safe="")
    credentials = f"{username}:{MASKED_PASSWORD}@"
    netloc = f"{credentials}{hostname}"
    if parts.port is not None:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def safe_error_message(error: BaseException, secrets: Iterable[str] = ()) -> str:
    message = str(error).strip() or error.__class__.__name__
    for secret in secrets:
        secret_text = str(secret or "").strip()
        if secret_text:
            message = message.replace(secret_text, mask_stream_url(secret_text))
    return message[:1000]


def normalize_local_file(value: str) -> str:
    path = Path(str(value).strip().strip('"'))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def parse_enabled(value: Any) -> bool:
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
    raise ChannelValidationError("启用状态必须是布尔值")


def source_key(source_type: str, camera_index_code: str, source_url: str) -> str:
    if source_type == SOURCE_TYPE_CAMERA_INDEX_CODE:
        return f"camera_index_code:{camera_index_code.strip()}"
    if source_type == SOURCE_TYPE_LOCAL_FILE:
        return f"local_file:{normalize_local_file(source_url).casefold()}"
    return f"rtsp:{source_url.strip()}"


def validate_channel_values(values: Dict[str, Any]) -> Dict[str, Any]:
    name = str(values.get("name", "")).strip()
    if not name:
        raise ChannelValidationError("通道名称不能为空")
    if len(name) > 100:
        raise ChannelValidationError("通道名称不能超过 100 个字符")

    source_type = str(values.get("source_type", "")).strip()
    if source_type not in SOURCE_TYPES:
        raise ChannelValidationError("视频源类型无效")

    camera_index_code = str(values.get("camera_index_code", "") or "").strip()
    raw_source_url = str(values.get("source_url", "") or "").strip().strip('"')
    camera_id = str(values.get("camera_id", "") or "").strip()
    enabled = parse_enabled(values.get("enabled", True))
    stream_profile = str(values.get("stream_profile", STREAM_PROFILE_MAIN) or "").strip().lower()
    if stream_profile not in STREAM_PROFILES:
        raise ChannelValidationError("码流类型必须是主码流或子码流")
    if source_type != SOURCE_TYPE_CAMERA_INDEX_CODE:
        stream_profile = STREAM_PROFILE_MAIN

    if source_type == SOURCE_TYPE_CAMERA_INDEX_CODE:
        if not camera_index_code:
            raise ChannelValidationError("cameraIndexCode 不能为空")
        raw_source_url = ""
    elif source_type == SOURCE_TYPE_RTSP:
        if not raw_source_url:
            raise ChannelValidationError("RTSP 地址不能为空")
        if not raw_source_url.lower().startswith("rtsp://"):
            raise ChannelValidationError("RTSP 地址必须以 rtsp:// 开头")
        camera_index_code = ""
    else:
        if not raw_source_url:
            raise ChannelValidationError("本地视频文件路径不能为空")
        raw_source_url = normalize_local_file(raw_source_url)
        suffix = Path(raw_source_url).suffix.lower()
        if suffix not in set(config.VIDEO_EXTENSIONS):
            raise ChannelValidationError(f"不支持的视频文件类型：{suffix or '无后缀'}")
        camera_index_code = ""

    return {
        "name": name,
        "source_type": source_type,
        "camera_index_code": camera_index_code,
        "source_url": raw_source_url,
        "camera_id": camera_id,
        "enabled": enabled,
        "stream_profile": stream_profile,
        "source_key": source_key(source_type, camera_index_code, raw_source_url),
    }


class VideoChannelStore:
    """SQLite-backed video channel store with one-time config import."""

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
                CREATE TABLE IF NOT EXISTS video_channels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    source_type TEXT NOT NULL,
                    camera_index_code TEXT NOT NULL DEFAULT '',
                    source_url TEXT NOT NULL DEFAULT '',
                    source_key TEXT NOT NULL UNIQUE,
                    camera_id TEXT NOT NULL DEFAULT '',
                    stream_profile TEXT NOT NULL DEFAULT 'main',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_status TEXT NOT NULL DEFAULT 'untested',
                    last_error TEXT NOT NULL DEFAULT '',
                    last_tested_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(video_channels)").fetchall()
            }
            if "stream_profile" not in columns:
                connection.execute(
                    "ALTER TABLE video_channels ADD COLUMN stream_profile TEXT NOT NULL DEFAULT 'main'"
                )
        self.import_configured_camera_index_codes()

    @staticmethod
    def _row_to_private(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["enabled"] = bool(data["enabled"])
        data["reconnect_enabled"] = data["source_type"] == SOURCE_TYPE_CAMERA_INDEX_CODE
        return data

    @staticmethod
    def to_public(channel: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(channel)
        data["source_url"] = mask_stream_url(str(data.get("source_url", "")))
        data["has_stream_credentials"] = data["source_url"] != str(channel.get("source_url", ""))
        data.pop("source_key", None)
        return data

    def list_channels(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM video_channels ORDER BY id").fetchall()
        return [self.to_public(self._row_to_private(row)) for row in rows]

    def get_channel(self, channel_id: int, public: bool = True) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM video_channels WHERE id = ?",
                (int(channel_id),),
            ).fetchone()
        if row is None:
            raise ChannelNotFoundError("视频通道不存在")
        channel = self._row_to_private(row)
        return self.to_public(channel) if public else channel

    def _unique_name(self, connection: sqlite3.Connection, preferred: str) -> str:
        candidate = preferred
        suffix = 2
        while connection.execute(
            "SELECT 1 FROM video_channels WHERE name = ? COLLATE NOCASE",
            (candidate,),
        ).fetchone():
            candidate = f"{preferred} ({suffix})"
            suffix += 1
        return candidate

    def import_configured_camera_index_codes(self) -> int:
        codes = [
            str(value).strip()
            for value in getattr(config, "VIDEO_STREAM_CAMERA_INDEX_CODES", [])
            if str(value).strip()
        ]
        display_names = dict(getattr(config, "CAMERA_DISPLAY_NAMES", {}))
        imported = 0
        with self._lock, self._connect() as connection:
            for index, code in enumerate(codes, start=1):
                key = source_key(SOURCE_TYPE_CAMERA_INDEX_CODE, code, "")
                if connection.execute(
                    "SELECT 1 FROM video_channels WHERE source_key = ?",
                    (key,),
                ).fetchone():
                    continue
                preferred_name = str(display_names.get(code) or f"Camera {index:02d}").strip()
                name = self._unique_name(connection, preferred_name)
                timestamp = now_text()
                connection.execute(
                    """
                    INSERT INTO video_channels (
                        name, source_type, camera_index_code, source_url, source_key,
                        camera_id, stream_profile, enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, '', ?, '', 'main', 1, ?, ?)
                    """,
                    (name, SOURCE_TYPE_CAMERA_INDEX_CODE, code, key, timestamp, timestamp),
                )
                imported += 1
        return imported

    def create_channel(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        values = validate_channel_values(payload)
        timestamp = now_text()
        try:
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO video_channels (
                        name, source_type, camera_index_code, source_url, source_key,
                        camera_id, stream_profile, enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        values["name"],
                        values["source_type"],
                        values["camera_index_code"],
                        values["source_url"],
                        values["source_key"],
                        values["camera_id"],
                        values["stream_profile"],
                        int(values["enabled"]),
                        timestamp,
                        timestamp,
                    ),
                )
                channel_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ChannelValidationError("通道名称或视频源已经存在") from exc
        return self.get_channel(channel_id)

    def update_channel(self, channel_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_channel(channel_id, public=False)
        merged = dict(current)
        merged.update(payload)
        supplied_url = str(payload.get("source_url", "")) if "source_url" in payload else None
        if supplied_url is not None and MASKED_PASSWORD in supplied_url:
            merged["source_url"] = current["source_url"]
        values = validate_channel_values(merged)
        try:
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE video_channels
                    SET name = ?, source_type = ?, camera_index_code = ?, source_url = ?,
                        source_key = ?, camera_id = ?, stream_profile = ?, enabled = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        values["name"],
                        values["source_type"],
                        values["camera_index_code"],
                        values["source_url"],
                        values["source_key"],
                        values["camera_id"],
                        values["stream_profile"],
                        int(values["enabled"]),
                        now_text(),
                        int(channel_id),
                    ),
                )
                if cursor.rowcount == 0:
                    raise ChannelNotFoundError("视频通道不存在")
        except sqlite3.IntegrityError as exc:
            raise ChannelValidationError("通道名称或视频源已经存在") from exc
        return self.get_channel(channel_id)

    def set_enabled(self, channel_id: int, enabled: bool) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE video_channels SET enabled = ?, updated_at = ? WHERE id = ?",
                (int(bool(enabled)), now_text(), int(channel_id)),
            )
            if cursor.rowcount == 0:
                raise ChannelNotFoundError("视频通道不存在")
        return self.get_channel(channel_id)

    def delete_channel(self, channel_id: int) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM video_channels WHERE id = ?",
                (int(channel_id),),
            )
            if cursor.rowcount == 0:
                raise ChannelNotFoundError("视频通道不存在")

    def record_test(self, channel_id: int, status: str, error: str = "") -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE video_channels
                SET last_status = ?, last_error = ?, last_tested_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, error, now_text(), now_text(), int(channel_id)),
            )
            if cursor.rowcount == 0:
                raise ChannelNotFoundError("视频通道不存在")
        return self.get_channel(channel_id)


def channel_source(channel: Dict[str, Any], make_camera_source: Callable[[str], str]) -> str:
    source_type = channel["source_type"]
    if source_type == SOURCE_TYPE_CAMERA_INDEX_CODE:
        source = make_camera_source(channel["camera_index_code"])
        stream_type = "1" if channel.get("stream_profile") == STREAM_PROFILE_SUB else "0"
        delimiter = "&" if "?" in source else "?"
        return f"{source}{delimiter}streamType={stream_type}"
    return channel["source_url"]


def probe_channel(
    channel: Dict[str, Any],
    resolve_capture_source: Callable[[str, Optional[float]], Tuple[Any, str]],
    make_camera_source: Callable[[str], str],
    timeout_seconds: float = CONNECTION_TEST_TIMEOUT_SECONDS,
    need_snapshot: bool = False,
) -> Tuple[Dict[str, Any], Optional[bytes]]:
    """Open and read a frame. cameraIndexCode sources retry by resolving a new URL."""
    started_at = time.monotonic()
    deadline = started_at + max(1.0, float(timeout_seconds))
    attempts = 3 if channel["source_type"] == SOURCE_TYPE_CAMERA_INDEX_CODE else 1
    source = channel_source(channel, make_camera_source)
    last_error = "视频连接失败"

    for attempt in range(1, attempts + 1):
        cap: Optional[cv2.VideoCapture] = None
        try:
            remaining_seconds = max(0.1, deadline - time.monotonic())
            capture_source, _resolved = resolve_capture_source(source, remaining_seconds)
            remaining_milliseconds = max(1000, int((deadline - time.monotonic()) * 1000))
            cap = cv2.VideoCapture()
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    min(int(config.VIDEO_OPEN_TIMEOUT_MILLISECONDS), remaining_milliseconds),
                )
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    min(int(config.VIDEO_READ_TIMEOUT_MILLISECONDS), remaining_milliseconds),
                )
            cap.open(capture_source)
            if not cap.isOpened():
                raise RuntimeError("无法打开视频源")

            frame = None
            read_attempts = 0
            while time.monotonic() < deadline and read_attempts < 30:
                read_attempts += 1
                ok, candidate = cap.read()
                if ok and candidate is not None:
                    frame = candidate
                    break
                time.sleep(0.03)
            if frame is None:
                raise RuntimeError("视频已连接，但在超时时间内没有读取到有效画面")

            height, width = frame.shape[:2]
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            snapshot_bytes: Optional[bytes] = None
            if need_snapshot:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(config.JPEG_QUALITY)],
                )
                if not ok:
                    raise RuntimeError("抓拍图片编码失败")
                snapshot_bytes = encoded.tobytes()
            result = {
                "success": True,
                "status": "online",
                "attempts": attempt,
                "reconnect_enabled": channel["source_type"] == SOURCE_TYPE_CAMERA_INDEX_CODE,
                "width": int(width),
                "height": int(height),
                "fps": round(fps, 2),
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            }
            return result, snapshot_bytes
        except Exception as exc:
            last_error = safe_error_message(exc, [str(channel.get("source_url", ""))])
        finally:
            if cap is not None:
                cap.release()

        if attempt < attempts and time.monotonic() < deadline:
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    raise RuntimeError(last_error)


def probe_channel_bounded(
    channel: Dict[str, Any],
    resolve_capture_source: Callable[[str, Optional[float]], Tuple[Any, str]],
    make_camera_source: Callable[[str], str],
    timeout_seconds: float = CONNECTION_TEST_TIMEOUT_SECONDS,
    need_snapshot: bool = False,
) -> Tuple[Dict[str, Any], Optional[bytes]]:
    """Guarantee an API response deadline even when an OpenCV backend ignores its timeout properties."""
    finished = threading.Event()
    outcome: Dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["value"] = probe_channel(
                channel,
                resolve_capture_source,
                make_camera_source,
                timeout_seconds=timeout_seconds,
                need_snapshot=need_snapshot,
            )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=worker, daemon=True, name="video-channel-probe").start()
    if not finished.wait(max(0.1, float(timeout_seconds))):
        raise TimeoutError(f"视频连接测试超时（{float(timeout_seconds):g} 秒）")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def register_video_channel_routes(
    app: Any,
    store: VideoChannelStore,
    resolve_capture_source: Callable[[str, Optional[float]], Tuple[Any, str]],
    make_camera_source: Callable[[str], str],
    delete_guard: Optional[Callable[[int], Optional[str]]] = None,
    shared_stream_manager: Any = None,
) -> None:
    """Attach page and JSON routes to a FastAPI application."""
    from fastapi import HTTPException, Request, Response
    from fastapi.responses import HTMLResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def handle_store_error(exc: BaseException) -> None:
        if isinstance(exc, ChannelNotFoundError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, ChannelValidationError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise exc

    def shared_probe(channel_id: int, need_snapshot: bool) -> Tuple[Dict[str, Any], Optional[bytes]]:
        started_at = time.monotonic()
        channel = store.get_channel(channel_id, public=False)
        capture, _resolved = shared_stream_manager.subscribe(channel_id)
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("共享视频流在超时时间内没有返回有效画面")
            height, width = frame.shape[:2]
            snapshot_bytes = None
            if need_snapshot:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(config.JPEG_QUALITY)],
                )
                if not ok:
                    raise RuntimeError("抓拍图片编码失败")
                snapshot_bytes = encoded.tobytes()
            traffic = shared_stream_manager.traffic_status(channel_id)
            return {
                "success": True,
                "status": "online",
                "attempts": 1,
                "reconnect_enabled": channel["source_type"] == SOURCE_TYPE_CAMERA_INDEX_CODE,
                "width": int(width),
                "height": int(height),
                "fps": round(float(capture.get(cv2.CAP_PROP_FPS) or 0.0), 2),
                "bitrate_kbps": traffic["bitrate_kbps"],
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            }, snapshot_bytes
        finally:
            capture.release()

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/channels", status_code=302)

    @app.get("/channels", response_class=HTMLResponse, include_in_schema=False)
    def channels_page(request: Request) -> Any:
        return templates.TemplateResponse(
            request=request,
            name="channels.html",
            context={"page_title": "视频通道"},
        )

    @app.get("/api/video-channels")
    def list_video_channels() -> Dict[str, Any]:
        channels = store.list_channels()
        if shared_stream_manager is not None:
            for channel in channels:
                channel["traffic"] = shared_stream_manager.traffic_status(int(channel["id"]))
        return {"items": channels, "total": len(channels)}

    @app.post("/api/video-channels", status_code=201)
    def create_video_channel(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return store.create_channel(payload)
        except (ChannelNotFoundError, ChannelValidationError) as exc:
            handle_store_error(exc)
            raise AssertionError("unreachable")

    @app.put("/api/video-channels/{channel_id}")
    def update_video_channel(channel_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            channel = store.update_channel(channel_id, payload)
            if shared_stream_manager is not None:
                if channel["enabled"]:
                    shared_stream_manager.restart_channel(channel_id)
                else:
                    shared_stream_manager.stop_channel(channel_id)
            return channel
        except (ChannelNotFoundError, ChannelValidationError) as exc:
            handle_store_error(exc)
            raise AssertionError("unreachable")

    @app.patch("/api/video-channels/{channel_id}/enabled")
    def set_video_channel_enabled(channel_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "enabled" not in payload:
            raise HTTPException(status_code=400, detail="请求体需要 enabled 字段")
        try:
            channel = store.set_enabled(channel_id, parse_enabled(payload["enabled"]))
            if shared_stream_manager is not None:
                if channel["enabled"]:
                    shared_stream_manager.restart_channel(channel_id)
                else:
                    shared_stream_manager.stop_channel(channel_id)
            return channel
        except (ChannelNotFoundError, ChannelValidationError) as exc:
            handle_store_error(exc)
            raise AssertionError("unreachable")

    @app.delete("/api/video-channels/{channel_id}")
    def delete_video_channel(channel_id: int) -> Dict[str, bool]:
        if delete_guard is not None:
            block_reason = delete_guard(channel_id)
            if block_reason:
                raise HTTPException(status_code=409, detail=block_reason)
        try:
            if shared_stream_manager is not None:
                shared_stream_manager.stop_channel(channel_id)
            store.delete_channel(channel_id)
        except ChannelNotFoundError as exc:
            handle_store_error(exc)
        return {"success": True}

    @app.post("/api/video-channels/{channel_id}/test")
    def test_video_channel(channel_id: int) -> Dict[str, Any]:
        try:
            channel = store.get_channel(channel_id, public=False)
            if shared_stream_manager is not None:
                result, _snapshot = shared_probe(channel_id, need_snapshot=False)
            else:
                result, _snapshot = probe_channel_bounded(
                    channel,
                    resolve_capture_source,
                    make_camera_source,
                    timeout_seconds=CONNECTION_TEST_TIMEOUT_SECONDS,
                    need_snapshot=False,
                )
            store.record_test(channel_id, "online")
            return result
        except ChannelNotFoundError as exc:
            handle_store_error(exc)
            raise AssertionError("unreachable")
        except Exception as exc:
            message = safe_error_message(exc)
            try:
                store.record_test(channel_id, "offline", message)
            except ChannelNotFoundError:
                pass
            raise HTTPException(status_code=502, detail=message) from exc

    @app.post("/api/video-channels/{channel_id}/snapshot")
    def snapshot_video_channel(channel_id: int) -> Response:
        try:
            channel = store.get_channel(channel_id, public=False)
            if shared_stream_manager is not None:
                _result, snapshot = shared_probe(channel_id, need_snapshot=True)
            else:
                _result, snapshot = probe_channel_bounded(
                    channel,
                    resolve_capture_source,
                    make_camera_source,
                    timeout_seconds=CONNECTION_TEST_TIMEOUT_SECONDS,
                    need_snapshot=True,
                )
            store.record_test(channel_id, "online")
            if snapshot is None:
                raise RuntimeError("抓拍未返回图片")
            return Response(content=snapshot, media_type="image/jpeg")
        except ChannelNotFoundError as exc:
            handle_store_error(exc)
            raise AssertionError("unreachable")
        except Exception as exc:
            message = safe_error_message(exc)
            try:
                store.record_test(channel_id, "offline", message)
            except ChannelNotFoundError:
                pass
            raise HTTPException(status_code=502, detail=message) from exc
