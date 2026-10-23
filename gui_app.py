"""Desktop operation UI for video preview, config editing, and alarm image push."""

from __future__ import annotations

import ast
import importlib
import json
import os
import queue
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import config
import push_alarm
from algorithms.machinery_hazard import defaults as machinery_defaults


os.environ.setdefault("YOLO_CONFIG_DIR", str(config.YOLO_CONFIG_DIR))
if getattr(config, "OPENCV_FFMPEG_CAPTURE_OPTIONS", ""):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = str(config.OPENCV_FFMPEG_CAPTURE_OPTIONS)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.py"
ALGORITHM_CONFIG_PATH = PROJECT_ROOT / "algorithms" / "machinery_hazard" / "defaults.py"
GUI_STATE_PATH = PROJECT_ROOT / "gui_state.json"
VIDEO_BOX_SIZE = (860, 484)
ALARM_PREVIEW_SIZE = (760, 560)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CONFIG_FIELDS = [
    ("VIDEO_STREAM_CAMERA_INDEX_CODES", "摄像头 cameraIndexCode"),
    ("VIDEO_STREAM_SOURCES", "直接视频流地址"),
    ("DETECTION_TIME_RANGES", "检测时间段"),
    ("DETECTION_INTERVAL_SECONDS", "检测间隔秒"),
    ("CONFIDENCE_THRESHOLD", "机械/全局置信度"),
    ("PERSON_CONFIDENCE_THRESHOLD", "person 置信度"),
    ("IOU_THRESHOLD", "NMS IoU 阈值"),
    ("IMAGE_SIZE", "推理尺寸"),
    ("MAX_DETECTIONS", "最大检测目标数"),
    ("MACHINERY_MOTION_ENABLED", "启用机械运动过滤"),
    ("MACHINERY_TRACK_IOU_THRESHOLD", "机械轨迹关联 IoU"),
    ("MACHINERY_TRACK_MAX_CENTER_DISTANCE_RATIO", "机械轨迹中心距离"),
    ("MACHINERY_TRACK_LOST_SECONDS", "机械轨迹丢失秒"),
    ("MACHINERY_BOX_SMOOTHING_ALPHA", "机械框平滑系数"),
    ("MACHINERY_MOTION_WINDOW_SECONDS", "车身运动窗口秒"),
    ("MACHINERY_MOTION_MIN_SAMPLES", "车身最少样本数"),
    ("MACHINERY_BODY_STEP_THRESHOLD_RATIO", "车身单步位移阈值"),
    ("MACHINERY_MOVING_THRESHOLD_RATIO", "车身累计位移阈值"),
    ("MACHINERY_MOVING_CONFIRM_COUNT", "车身移动确认次数"),
    ("MACHINERY_STOP_CONFIRM_SECONDS", "车身停止确认秒"),
    ("ARM_MOTION_CLASS_IDS", "机械臂检测类别"),
    ("ARM_FLOW_INTERVAL_SECONDS", "机械臂光流间隔秒"),
    ("ARM_ROI_EXPAND_RATIO", "机械臂 ROI 扩大比例"),
    ("ARM_BODY_EXCLUDE_BOTTOM_RATIO", "光流车身排除比例"),
    ("ARM_PERSON_MASK_EXPAND_RATIO", "光流人员遮罩扩大"),
    ("ARM_FLOW_ROI_MAX_SIZE", "光流 ROI 最大尺寸"),
    ("ARM_FLOW_MIN_VALID_PIXELS", "光流最少有效像素"),
    ("ARM_FLOW_MIN_SPEED_RATIO", "机械臂光流速度阈值"),
    ("ARM_FLOW_MIN_MOVING_PIXEL_RATIO", "光流运动像素比例"),
    ("ARM_FLOW_MIN_COMPONENT_RATIO", "光流连续区域比例"),
    ("ARM_FLOW_WINDOW_SIZE", "机械臂投票窗口"),
    ("ARM_FLOW_CONFIRM_COUNT", "机械臂确认次数"),
    ("ARM_FLOW_STOP_SECONDS", "机械臂停止确认秒"),
    ("ARM_FLOW_MAX_BRIGHTNESS_DELTA", "光流亮度变化上限"),
    ("ARM_GLOBAL_MOTION_COMPENSATION", "摄像头运动补偿"),
    ("ARM_GLOBAL_MOTION_MIN_POINTS", "背景最少特征点"),
    ("LOG_MACHINERY_MOTION", "打印机械运动日志"),
    ("RIDER_FILTER_ENABLED", "过滤乘坐人员"),
    ("RIDER_FILTER_CLASS_IDS", "乘坐过滤机械类别"),
    ("RIDER_CABIN_ZONES", "各车型驾驶区域"),
    ("RIDER_DEFAULT_CABIN_ZONE", "默认驾驶区域"),
    ("RIDER_PERSON_CONTAINMENT_THRESHOLD", "乘员框包含比例"),
    ("RIDER_PERSON_BOTTOM_MAX_RATIO", "乘员底部高度上限"),
    ("RIDER_RELATIVE_POSITION_THRESHOLD", "乘员相对位移阈值"),
    ("RIDER_CONFIRM_WINDOW_SIZE", "乘员确认窗口"),
    ("RIDER_CONFIRM_COUNT", "乘员确认次数"),
    ("RIDER_EXIT_SECONDS", "乘员退出确认秒"),
    ("PERSON_TRACK_IOU_THRESHOLD", "人员轨迹关联 IoU"),
    ("PERSON_TRACK_MAX_CENTER_DISTANCE_RATIO", "人员轨迹中心距离"),
    ("PERSON_TRACK_LOST_SECONDS", "人员轨迹丢失秒"),
    ("RISK_ZONE_CLASS_SCALES", "各车型椭圆范围"),
    ("RISK_ZONE_CENTER_Y_RATIO", "椭圆中心高度"),
    ("RISK_ZONE_USE_MOTION_DIRECTION", "椭圆跟随运动方向"),
    ("RISK_ZONE_REACTION_TIME_SECONDS", "椭圆反应时间秒"),
    ("RISK_ZONE_MAX_FORWARD_EXTENSION_RATIO", "椭圆最大前向延伸"),
    ("PERSON_RISK_LOWER_BODY_RATIO", "人员下部检测比例"),
    ("BUFFER_SHAPE", "危险区形状"),
    ("BUFFER_SCALE", "危险区放大系数"),
    ("OVERLAP_RATIO_THRESHOLD", "重合报警阈值"),
    ("RISK_CONSECUTIVE_FRAMES", "连续报警帧数"),
    ("ALARM_SAVE_COOLDOWN_SECONDS", "报警保存冷却秒"),
    ("CAMERA_DISPLAY_NAMES", "报警图 Camera 名称"),
    ("ALARM_SAVE_ANNOTATED_IMAGE", "保存标注图"),
    ("ALARM_SAVE_ORIGINAL_IMAGE", "保存原图"),
    ("ALARM_SAVE_RISK_CROP", "保存局部图"),
    ("ALARM_SAVE_METADATA_JSON", "保存事件 JSON"),
    ("ALARM_SAVE_SHORT_VIDEO", "保存报警短视频"),
    ("ALARM_CROP_PADDING_RATIO", "局部图扩大比例"),
    ("ALARM_CLIP_PRE_SECONDS", "报警前视频秒"),
    ("ALARM_CLIP_POST_SECONDS", "报警后视频秒"),
    ("ALARM_CLIP_MAX_FPS", "报警视频最高 FPS"),
    ("ALARM_RETENTION_MODE", "报警留存模式"),
    ("ALARM_RETENTION_DAYS", "报警保留天数"),
    ("ALARM_RETENTION_MAX_EVENTS", "报警保留事件数"),
    ("ALARM_RISK_TEXT", "报警红字内容"),
    ("ALARM_TITLE_FONT_SIZE", "报警标题字号"),
    ("ALARM_TITLE_TOP_MARGIN", "报警标题上边距"),
    ("ALARM_RISK_BOX_ALPHA", "风险框填充透明度"),
    ("ALARM_RISK_BOX_BORDER_THICKNESS", "风险框边线宽度"),
    ("ALARM_DRAW_DEDUP_IOU_THRESHOLD", "重复框 IoU 阈值"),
    ("ALARM_DRAW_DEDUP_CONTAINMENT_THRESHOLD", "重复框包含阈值"),
    ("ALARM_IMAGE_EXT", "报警图片后缀"),
    ("JPEG_QUALITY", "JPG 质量"),
    ("DRAW_LINE_THICKNESS", "检测框线宽"),
    ("DRAW_PERSON_BOX", "绘制人员框"),
    ("DRAW_MACHINERY_BOX", "绘制机械框"),
    ("DRAW_LABELS", "绘制类别置信度"),
    ("SHOW_WINDOW", "命令行显示窗口"),
    ("SAVE_ANNOTATED_VIDEO", "保存标注视频"),
]


def config_value(name: str) -> Any:
    if hasattr(machinery_defaults, name):
        return getattr(machinery_defaults, name)
    return getattr(config, name)


def value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def parse_field_value(text: str, current_value: Any) -> Any:
    stripped = text.strip()
    if isinstance(current_value, bool):
        lowered = stripped.lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"布尔值只能填写 True 或 False：{text}")
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(stripped)
    if isinstance(current_value, float):
        return float(stripped)
    if isinstance(current_value, str):
        return text
    return ast.literal_eval(stripped)


def format_assignment_value(value: Any) -> str:
    return repr(value)


def update_assignment_file(path: Path, values: Dict[str, Any]) -> None:
    if not values:
        return
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    nodes: Dict[str, ast.Assign] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    nodes[target.id] = node

    replacements = []
    for name, value in values.items():
        node = nodes.get(name)
        if node is None or node.end_lineno is None:
            raise KeyError(f"{path.name} 中未找到配置项：{name}")
        replacements.append((node.lineno - 1, node.end_lineno, f"{name} = {format_assignment_value(value)}\n"))

    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]

    path.write_text("".join(lines), encoding="utf-8")


def update_config_file(values: Dict[str, Any]) -> None:
    algorithm_values = {name: value for name, value in values.items() if hasattr(machinery_defaults, name)}
    platform_values = {name: value for name, value in values.items() if name not in algorithm_values}
    update_assignment_file(CONFIG_PATH, platform_values)
    update_assignment_file(ALGORITHM_CONFIG_PATH, algorithm_values)


def reload_configuration_modules() -> None:
    importlib.reload(machinery_defaults)
    importlib.reload(config)


def extract_rtsp_url(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, dict):
        for key in ("data", "url", "rtsp", "streamUrl", "stream_url"):
            found = extract_rtsp_url(value.get(key))
            if found:
                return found
        for item in value.values():
            found = extract_rtsp_url(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = extract_rtsp_url(item)
            if found:
                return found
    return None


def fetch_rtsp_url_by_camera_index_code(camera_index_code: str) -> str:
    payload = dict(getattr(config, "VIDEO_STREAM_API_PAYLOAD", {}))
    payload["cameraIndexCode"] = camera_index_code
    request = urllib.request.Request(
        str(config.VIDEO_STREAM_API_URL),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = float(getattr(config, "VIDEO_STREAM_API_TIMEOUT_SECONDS", 10.0))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))

    rtsp_url = extract_rtsp_url(data)
    if not rtsp_url:
        raise RuntimeError(f"接口未返回 RTSP 地址：{data}")
    return rtsp_url


def configured_video_sources() -> List[str]:
    sources = [str(item) for item in getattr(config, "VIDEO_STREAM_SOURCES", []) if str(item).strip()]
    sources.extend(
        f"cameraIndexCode:{item}"
        for item in getattr(config, "VIDEO_STREAM_CAMERA_INDEX_CODES", [])
        if str(item).strip()
    )
    return sources


def read_image_with_unicode_path(path: Path) -> Any:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图片读取失败：{path}")
    return image


class VideoReader(threading.Thread):
    def __init__(self, app: "OperationApp", source: str) -> None:
        super().__init__(daemon=True)
        self.app = app
        self.source = source
        self.stop_event = threading.Event()
        self.cap: Optional[cv2.VideoCapture] = None

    def stop(self) -> None:
        self.stop_event.set()
        if self.cap is not None:
            self.cap.release()

    def run(self) -> None:
        capture_source = self.source
        try:
            if capture_source.startswith("cameraIndexCode:"):
                code = capture_source.split(":", 1)[1].strip()
                self.app.post_status("正在解析 cameraIndexCode...")
                capture_source = fetch_rtsp_url_by_camera_index_code(code)

            self.app.post_status("正在打开视频流...")
            self.cap = cv2.VideoCapture()
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(config.VIDEO_OPEN_TIMEOUT_MILLISECONDS))
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(config.VIDEO_READ_TIMEOUT_MILLISECONDS))
            self.cap.open(capture_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频流：{capture_source}")

            self.app.post_status("视频流已打开")
            failed_reads = 0
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    failed_reads += 1
                    if failed_reads >= 30:
                        raise RuntimeError("连续读取视频帧失败")
                    time.sleep(0.05)
                    continue
                failed_reads = 0
                self.app.set_latest_frame(frame)
                time.sleep(0.01)
        except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
            self.app.post_status(f"视频错误：{exc}")
        except Exception as exc:
            self.app.post_status(f"视频异常：{exc}")
        finally:
            if self.cap is not None:
                self.cap.release()
            self.app.post_video_stopped()


class OperationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("工程机械风险检测操作台")
        self.root.geometry("1280x760")
        self.root.minsize(1080, 680)

        self.status_queue: "queue.Queue[str]" = queue.Queue()
        self.video_reader: Optional[VideoReader] = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.photo = None
        self.alarm_preview_photo = None
        self.alarm_preview_paths: List[Path] = []
        self.marked_for_push_paths = self.load_marked_for_push_paths()
        self.config_entries: Dict[str, tk.Widget] = {}

        self._build_ui()
        self.refresh_video_sources()
        self.load_config_fields()
        self.root.after(50, self.render_video_frame)
        self.root.after(100, self.poll_status)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_marked_for_push_paths(self) -> set[str]:
        if not GUI_STATE_PATH.exists():
            return set()
        try:
            data = json.loads(GUI_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return set()
        return {str(item) for item in data.get("marked_for_push_paths", [])}

    def save_marked_for_push_paths(self) -> None:
        data = {"marked_for_push_paths": sorted(self.marked_for_push_paths)}
        GUI_STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def alarm_path_key(self, path: Path) -> str:
        return str(path.resolve())

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._build_video_panel(notebook)
        self._build_config_panel(notebook)
        self._build_alarm_push_panel(notebook)
        self._build_alarm_preview_panel(notebook)

        self.status_var = tk.StringVar(value="就绪")
        status = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W)
        status.pack(fill=tk.X, padx=10, pady=(0, 8))

    def _build_video_panel(self, notebook: ttk.Notebook) -> None:
        parent = ttk.Frame(notebook, padding=10)
        notebook.add(parent, text="视频流浏览")

        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(controls, text="视频源").pack(side=tk.LEFT)
        self.source_var = tk.StringVar()
        self.source_combo = ttk.Combobox(controls, textvariable=self.source_var, width=58)
        self.source_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        ttk.Button(controls, text="刷新", command=self.refresh_video_sources).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="开始预览", command=self.start_video).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="停止", command=self.stop_video).pack(side=tk.LEFT, padx=3)

        self.video_label = ttk.Label(parent, anchor=tk.CENTER, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.configure(text="视频预览")

    def _build_config_panel(self, notebook: ttk.Notebook) -> None:
        outer = ttk.Frame(notebook)
        notebook.add(outer, text="配置")

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        form = ttk.Frame(canvas)
        form.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for row, (name, label) in enumerate(CONFIG_FIELDS):
            ttk.Label(form, text=label, width=22).grid(row=row, column=0, sticky=tk.W, padx=8, pady=5)
            current = config_value(name)
            if isinstance(current, bool):
                var = tk.BooleanVar(value=current)
                widget = ttk.Checkbutton(form, variable=var)
                widget.var = var  # type: ignore[attr-defined]
            else:
                widget = ttk.Entry(form, width=48)
                widget.insert(0, value_to_text(current))
            widget.grid(row=row, column=1, sticky=tk.EW, padx=8, pady=5)
            self.config_entries[name] = widget
        form.columnconfigure(1, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=8)
        ttk.Button(buttons, text="重新加载配置", command=self.reload_config).pack(side=tk.LEFT, padx=8)
        ttk.Button(buttons, text="保存配置", command=self.save_config).pack(side=tk.RIGHT, padx=8)

    def _build_alarm_push_panel(self, notebook: ttk.Notebook) -> None:
        panel = ttk.Frame(notebook, padding=10)
        notebook.add(panel, text="报警推送")

        self.push_image_var = tk.StringVar(value=push_alarm.IMAGE_PATH)
        self.push_camera_var = tk.StringVar(value=push_alarm.CAMERA_ID)
        self.push_api_var = tk.StringVar(value=push_alarm.API_URL)
        self.push_dry_run_var = tk.BooleanVar(value=push_alarm.DRY_RUN)

        ttk.Label(panel, text="报警图片").grid(row=0, column=0, sticky=tk.W, pady=6)
        ttk.Entry(panel, textvariable=self.push_image_var).grid(row=0, column=1, sticky=tk.EW, pady=6)
        ttk.Button(panel, text="选择", command=self.choose_alarm_image).grid(row=0, column=2, padx=6, pady=6)

        ttk.Label(panel, text="CameraID").grid(row=1, column=0, sticky=tk.W, pady=6)
        ttk.Entry(panel, textvariable=self.push_camera_var).grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=6)

        ttk.Label(panel, text="接口地址").grid(row=2, column=0, sticky=tk.W, pady=6)
        ttk.Entry(panel, textvariable=self.push_api_var).grid(row=2, column=1, columnspan=2, sticky=tk.EW, pady=6)

        ttk.Checkbutton(panel, text="Dry run，只预览参数", variable=self.push_dry_run_var).grid(
            row=3, column=1, sticky=tk.W, pady=6
        )

        ttk.Button(panel, text="推送报警", command=self.push_alarm_image).grid(row=4, column=1, sticky=tk.E, pady=8)

        self.push_output = tk.Text(panel, height=16, wrap=tk.WORD)
        self.push_output.grid(row=5, column=0, columnspan=3, sticky=tk.NSEW, pady=8)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(5, weight=1)

    def _build_alarm_preview_panel(self, notebook: ttk.Notebook) -> None:
        panel = ttk.Frame(notebook, padding=10)
        notebook.add(panel, text="报警图片预览")

        toolbar = ttk.Frame(panel)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(toolbar, text="刷新图片", command=self.refresh_alarm_images).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="用于推送", command=self.use_selected_alarm_image).pack(side=tk.LEFT, padx=6)

        content = ttk.PanedWindow(panel, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(content)
        preview_frame = ttk.Frame(content)
        content.add(list_frame, weight=1)
        content.add(preview_frame, weight=3)

        self.alarm_image_list = tk.Listbox(list_frame, height=24)
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.alarm_image_list.yview)
        self.alarm_image_list.configure(yscrollcommand=list_scroll.set)
        self.alarm_image_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.alarm_image_list.bind("<<ListboxSelect>>", self.on_alarm_image_selected)
        self.alarm_image_list.bind("<Double-Button-1>", lambda _event: self.use_selected_alarm_image())

        self.alarm_preview_label = ttk.Label(preview_frame, anchor=tk.CENTER, background="black", text="选择报警图片预览")
        self.alarm_preview_label.pack(fill=tk.BOTH, expand=True)

        self.alarm_preview_info = tk.StringVar(value="")
        ttk.Label(preview_frame, textvariable=self.alarm_preview_info, anchor=tk.W).pack(fill=tk.X, pady=(6, 0))

        self.refresh_alarm_images()

    def refresh_video_sources(self) -> None:
        reload_configuration_modules()
        sources = configured_video_sources()
        self.source_combo["values"] = sources
        if sources and not self.source_var.get():
            self.source_var.set(sources[0])
        self.post_status(f"已加载 {len(sources)} 个视频源")

    def load_config_fields(self) -> None:
        for name, widget in self.config_entries.items():
            current = config_value(name)
            if isinstance(widget, ttk.Checkbutton):
                widget.var.set(bool(current))  # type: ignore[attr-defined]
            else:
                widget.delete(0, tk.END)
                widget.insert(0, value_to_text(current))

    def reload_config(self) -> None:
        reload_configuration_modules()
        self.load_config_fields()
        self.refresh_video_sources()
        self.post_status("配置已重新加载")

    def save_config(self) -> None:
        try:
            values: Dict[str, Any] = {}
            for name, widget in self.config_entries.items():
                current = config_value(name)
                if isinstance(widget, ttk.Checkbutton):
                    values[name] = bool(widget.var.get())  # type: ignore[attr-defined]
                else:
                    values[name] = parse_field_value(widget.get(), current)
            update_config_file(values)
            reload_configuration_modules()
            self.load_config_fields()
            self.refresh_video_sources()
            self.post_status("配置已保存")
            messagebox.showinfo(
                "保存成功",
                "平台配置和机械伤害算法配置已分别保存。检测服务需要重新启动后使用新配置。",
            )
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def start_video(self) -> None:
        source = self.source_var.get().strip()
        if not source:
            messagebox.showwarning("缺少视频源", "请先选择或输入视频源。")
            return
        self.stop_video()
        self.video_reader = VideoReader(self, source)
        self.video_reader.start()

    def stop_video(self) -> None:
        if self.video_reader is not None:
            self.video_reader.stop()
            self.video_reader = None
        self.post_status("视频预览已停止")

    def set_latest_frame(self, frame: Any) -> None:
        with self.frame_lock:
            self.latest_frame = frame.copy()

    def render_video_frame(self) -> None:
        frame = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = self.resize_for_preview(rgb)
            height, width = rgb.shape[:2]
            ppm = b"P6\n%d %d\n255\n" % (width, height) + rgb.tobytes()
            self.photo = tk.PhotoImage(data=ppm, format="PPM")
            self.video_label.configure(image=self.photo, text="")

        self.root.after(33, self.render_video_frame)

    def resize_for_preview(self, frame: Any) -> Any:
        return self.resize_image_for_box(frame, VIDEO_BOX_SIZE)

    def resize_image_for_box(self, frame: Any, box_size: tuple[int, int]) -> Any:
        max_width, max_height = box_size
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1.0:
            return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        return frame

    def choose_alarm_image(self) -> None:
        initial_dir = PROJECT_ROOT / "alarm_outputs"
        path = filedialog.askopenfilename(
            initialdir=str(initial_dir if initial_dir.exists() else PROJECT_ROOT),
            title="选择报警图片",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files", "*.*")],
        )
        if path:
            self.push_image_var.set(path)

    def refresh_alarm_images(self) -> None:
        alarm_dir = PROJECT_ROOT / "alarm_outputs"
        self.alarm_preview_paths = []
        self.alarm_image_list.delete(0, tk.END)

        if not alarm_dir.exists():
            self.alarm_preview_info.set("报警图片目录不存在")
            return

        files = [
            path
            for path in alarm_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        self.alarm_preview_paths = files

        for path in files:
            push_tag = "[已选] " if self.alarm_path_key(path) in self.marked_for_push_paths else ""
            if path.stem == "alarm":
                artifact_tag = "  [标注图]"
            elif path.stem == "original" or "_original" in path.stem or path.parent.name == "original":
                artifact_tag = "  [原图]"
            elif path.stem == "risk_crop":
                artifact_tag = "  [局部图]"
            else:
                artifact_tag = ""
            display_name = str(path.relative_to(alarm_dir))
            self.alarm_image_list.insert(tk.END, f"{push_tag}{display_name}{artifact_tag}")

        self.alarm_preview_info.set(f"共 {len(files)} 张报警图片")
        if files:
            self.alarm_image_list.selection_set(0)
            self.preview_alarm_image(files[0])

    def selected_alarm_image_path(self) -> Optional[Path]:
        selection = self.alarm_image_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if index < 0 or index >= len(self.alarm_preview_paths):
            return None
        return self.alarm_preview_paths[index]

    def on_alarm_image_selected(self, _event: Any = None) -> None:
        path = self.selected_alarm_image_path()
        if path is not None:
            self.preview_alarm_image(path)

    def preview_alarm_image(self, path: Path) -> None:
        try:
            image = read_image_with_unicode_path(path)
        except Exception as exc:
            self.alarm_preview_label.configure(image="", text="图片读取失败")
            self.alarm_preview_info.set(f"{path} | {exc}")
            return

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = self.resize_image_for_box(rgb, ALARM_PREVIEW_SIZE)
        height, width = rgb.shape[:2]
        ppm = b"P6\n%d %d\n255\n" % (width, height) + rgb.tobytes()
        self.alarm_preview_photo = tk.PhotoImage(data=ppm, format="PPM")
        self.alarm_preview_label.configure(image=self.alarm_preview_photo, text="")

        size_mb = path.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        self.alarm_preview_info.set(f"{path.name} | {size_mb:.2f} MB | {mtime}")

    def use_selected_alarm_image(self) -> None:
        path = self.selected_alarm_image_path()
        if path is None:
            messagebox.showwarning("未选择图片", "请先在报警图片预览中选择一张图片。")
            return
        self.push_image_var.set(str(path))
        self.marked_for_push_paths.add(self.alarm_path_key(path))
        self.save_marked_for_push_paths()
        current_index = self.alarm_preview_paths.index(path) if path in self.alarm_preview_paths else None
        self.refresh_alarm_images()
        if current_index is not None and current_index < self.alarm_image_list.size():
            self.alarm_image_list.selection_clear(0, tk.END)
            self.alarm_image_list.selection_set(current_index)
            self.alarm_image_list.see(current_index)
        self.post_status(f"已选择用于推送：{path.name}")

    def push_alarm_image(self) -> None:
        thread = threading.Thread(target=self._push_alarm_image_worker, daemon=True)
        thread.start()

    def _push_alarm_image_worker(self) -> None:
        try:
            image_path = Path(self.push_image_var.get().strip())
            camera_id = self.push_camera_var.get().strip()
            api_url = self.push_api_var.get().strip()
            payload = push_alarm.build_payload(image_path, camera_id, datetime.now())

            if self.push_dry_run_var.get():
                preview = dict(payload)
                preview["ViolationPhoto"] = f"<base64 length={len(payload['ViolationPhoto'])}>"
                self.set_push_output(preview)
                self.post_status("报警推送 dry-run 完成")
                return

            result = push_alarm.push_alarm(api_url, payload, push_alarm.TIMEOUT_SECONDS)
            self.set_push_output(result)
            self.post_status("报警推送完成")
        except Exception as exc:
            self.set_push_output({"success": False, "error": str(exc)})
            self.post_status(f"报警推送失败：{exc}")

    def set_push_output(self, data: Dict[str, Any]) -> None:
        text = json.dumps(data, ensure_ascii=False, indent=2)
        self.root.after(0, self._replace_push_output, text)

    def _replace_push_output(self, text: str) -> None:
        self.push_output.delete("1.0", tk.END)
        self.push_output.insert(tk.END, text)

    def post_status(self, text: str) -> None:
        self.status_queue.put(text)

    def post_video_stopped(self) -> None:
        self.status_queue.put("视频预览线程已退出")

    def poll_status(self) -> None:
        while True:
            try:
                self.status_var.set(self.status_queue.get_nowait())
            except queue.Empty:
                break
        self.root.after(100, self.poll_status)

    def on_close(self) -> None:
        self.stop_video()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    OperationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
