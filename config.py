"""Platform-wide configuration shared by all detection algorithms.

Algorithm-specific defaults live below ``algorithms/<algorithm>/defaults.py``.
The module-level ``__getattr__`` at the bottom is a read-only compatibility
bridge for legacy code and can be removed after all callers use the registry.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_CONFIG_DIR = PROJECT_ROOT / ".ultralytics_settings"

# Configured sources and cameraIndexCode resolution.
DEFAULT_SOURCE = ""
IMAGE_SOURCES = []
VIDEO_STREAM_SOURCES = []
VIDEO_STREAM_API_URL = "http://58.49.74.232:64302/api/HkPlat/GetUrlBycameraIndexCode"
VIDEO_STREAM_CAMERA_INDEX_CODES = [
    "fa3df63d1ac14ef3b5d65922d31344c8",
    "9628b1037e45446f8fc1712c808eb81f",
    "9beba9b7836e49ab8ce1ffa4281b4829",
]
VIDEO_STREAM_API_PAYLOAD = {
    "cameraIndexCode": "",
    "streamType": "0",
    "protocol": "rtsp",
    "transmode": "1",
    "expand": "transcode=0",
    "streamform": "rtp",
}
VIDEO_STREAM_API_TIMEOUT_SECONDS = 10.0
OPENCV_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;tcp"

# Shared stream scheduling, capture, and reconnection.
PROCESS_VIDEO_STREAMS_IN_PARALLEL = True
VIDEO_STREAM_WORKERS = 0
DETECTION_TIME_RANGES = []
STREAM_MAX_RUNTIME_SECONDS = 0
VIDEO_RECONNECT_ENABLED = True
VIDEO_RECONNECT_INTERVAL_SECONDS = 5.0
VIDEO_RECONNECT_MAX_ATTEMPTS = 0
STREAM_WARMUP_FRAMES = 100
VIDEO_READ_FAIL_RECONNECT_FRAMES = 5
VIDEO_OPEN_TIMEOUT_MILLISECONDS = 10000
VIDEO_READ_TIMEOUT_MILLISECONDS = 10000

# Platform logging.
LOG_DETECTION_FRAMES = True
DETECTION_LOG_EVERY_N_FRAMES = 1
LOG_ALARMS = True
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_NAME = "service.log"
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5

# Channel display names used by task and alarm metadata.
CAMERA_DISPLAY_NAMES = {
    "fa3df63d1ac14ef3b5d65922d31344c8": "Camera 01",
    "9628b1037e45446f8fc1712c808eb81f": "Camera 02",
    "9beba9b7836e49ab8ce1ffa4281b4829": "Camera 03",
}
DEFAULT_CAMERA_DISPLAY_NAME = "Camera"

# Shared alarm evidence schema, encoding, and retention.
ALARM_OUTPUT_DIR = PROJECT_ROOT / "alarm_outputs"
ALARM_EVENT_DIR_PREFIX = "event"
ALARM_SAVE_ANNOTATED_IMAGE = True
ALARM_SAVE_ORIGINAL_IMAGE = True
ALARM_SAVE_RISK_CROP = True
ALARM_SAVE_METADATA_JSON = True
ALARM_SAVE_SHORT_VIDEO = True
ALARM_CROP_PADDING_RATIO = 0.15
ALARM_CLIP_PRE_SECONDS = 2.0
ALARM_CLIP_POST_SECONDS = 3.0
ALARM_CLIP_MAX_FPS = 10.0
ALARM_CLIP_CODEC = "mp4v"
ALARM_CLIP_JPEG_QUALITY = 85
ALARM_RETENTION_MODE = "count"
ALARM_RETENTION_DAYS = 7.0
ALARM_RETENTION_MAX_EVENTS = 2000
ALARM_FONT_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")
ALARM_INFO_FONT_SIZE = 20
ALARM_IMAGE_EXT = ".jpg"
JPEG_QUALITY = 95

# Command-line display and output behavior shared by algorithms.
SHOW_WINDOW = False
SAVE_ANNOTATED_VIDEO = False
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "video_outputs"
VIDEO_CODEC = "mp4v"

# Web service and supported local file types.
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}

# Local visual model training. Training runs in a separate process so the Web
# service remains responsive and production weights are never overwritten.
MODEL_TRAINING_ROOT = PROJECT_ROOT / "model_training_outputs"
MODEL_TRAINING_PYTHON = Path(r"C:\Users\LENOVO\anaconda3\envs\yolov5\python.exe")
MODEL_TRAINING_YOLOV5_REPO = Path(r"C:\Users\LENOVO\PycharmProjects\YOLOv5\yolov5")
MODEL_TRAINING_RUNNER = PROJECT_ROOT / "training_support" / "yolov5_runner.py"
MODEL_TRAINING_YOLO_CONFIG_DIR = Path(r"C:\Users\LENOVO\.ultralytics_settings_training")
SAFETY_HELMET_TRAINING_WEIGHTS = PROJECT_ROOT / "models" / "safety_helmet" / "model1_best.pt"
MODEL_TRAINING_DEFAULT_IMAGE_SIZE = 640
MODEL_TRAINING_DEFAULT_EPOCHS = 100
MODEL_TRAINING_DEFAULT_BATCH_SIZE = 2
MODEL_TRAINING_DEFAULT_WORKERS = 2
MODEL_TRAINING_DEFAULT_PATIENCE = 20
MODEL_TRAINING_RECOMMENDED_SAMPLES = 300
MODEL_TRAINING_MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024
MODEL_TRAINING_MAX_EXTRACTED_BYTES = 50 * 1024 * 1024 * 1024

# Local multimodal alarm review. New alarms are retained locally first, then
# reviewed serially before an external report is allowed to leave the device.
ALARM_REVIEW_ENABLED = True
ALARM_REVIEW_SERVER_URL = "http://127.0.0.1:8010/v1/chat/completions"
ALARM_REVIEW_SERVER_HOST = "127.0.0.1"
ALARM_REVIEW_SERVER_PORT = 8010
ALARM_REVIEW_MODEL_NAME = "Qwen3-VL-8B-Instruct-Q4_K_M"
ALARM_REVIEW_MODEL_PATH = Path(r"E:\AI\Models\Qwen3-VL-8B-Instruct\Qwen3VL-8B-Instruct-Q4_K_M.gguf")
ALARM_REVIEW_MMPROJ_PATH = Path(r"E:\AI\Models\Qwen3-VL-8B-Instruct\mmproj-Qwen3VL-8B-Instruct-F16.gguf")
ALARM_REVIEW_LLAMA_SERVER_PATH = Path(
    r"C:\Users\LENOVO\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
)
ALARM_REVIEW_PROMPT_DIR = PROJECT_ROOT / "review" / "prompts"
ALARM_REVIEW_KEYFRAME_COUNT = 5
ALARM_REVIEW_REQUEST_TIMEOUT_SECONDS = 180.0
ALARM_REVIEW_ASSET_WAIT_SECONDS = 30.0
ALARM_REVIEW_SERVER_START_TIMEOUT_SECONDS = 90.0
# Allow complete structured observations without spending the budget on thinking.
ALARM_REVIEW_MAX_TOKENS = 1536
ALARM_REVIEW_GPU_LAYERS = 28
ALARM_REVIEW_CONTEXT_SIZE = 8192
ALARM_REVIEW_IMAGE_MAX_TOKENS = 768


def __getattr__(name: str):
    """Temporarily expose migrated defaults to legacy modules as read-only values."""
    if name.isupper():
        from algorithms.machinery_hazard import defaults as machinery_defaults

        if hasattr(machinery_defaults, name):
            return getattr(machinery_defaults, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
