import atexit
import threading
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "captured_images"
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WARMUP_FRAMES = 5

_camera_lock = threading.Lock()
_camera_handle = None


def _open_camera():
    global _camera_handle

    if cv2 is None:
        return None, "OpenCV is not installed."

    if _camera_handle is not None and _camera_handle.isOpened():
        return _camera_handle, ""

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)

    if not cap.isOpened():
        cap.release()
        return None, f"Failed to open camera index {CAMERA_INDEX}."

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    _camera_handle = cap
    return _camera_handle, ""


def release_camera():
    global _camera_handle

    with _camera_lock:
        if _camera_handle is not None:
            _camera_handle.release()
            _camera_handle = None


def capture_image(prefix: str = "warning", output_path: str = "") -> dict:
    with _camera_lock:
        cap, error_message = _open_camera()

        if cap is None:
            return {
                "image_path": "",
                "captured_at": datetime.now().isoformat(timespec="seconds"),
                "camera_ok": False,
                "message": error_message,
            }

        frame = None
        ret = False

        for _ in range(WARMUP_FRAMES):
            ret, frame = cap.read()

        if not ret or frame is None:
            return {
                "image_path": "",
                "captured_at": datetime.now().isoformat(timespec="seconds"),
                "camera_ok": False,
                "message": "Failed to capture frame from camera.",
            }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_path = Path(output_path) if output_path else IMAGE_DIR / f"{prefix}_{timestamp}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(image_path), frame):
            return {
                "image_path": "",
                "captured_at": datetime.now().isoformat(timespec="seconds"),
                "camera_ok": False,
                "message": f"Failed to save image to {image_path}.",
            }

        return {
            "image_path": str(image_path),
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "camera_ok": True,
            "message": "",
        }


atexit.register(release_camera)
