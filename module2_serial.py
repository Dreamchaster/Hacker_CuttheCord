import copy
import csv
import json
import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import serial

from module3_llm import analyze_sensor_payload
from module4_camera import capture_image
from module5_vision import analyze_image

BASE_DIR = Path(__file__).parent

PORT = "COM9"
BAUDRATE = 115200

CSV_FILE = BASE_DIR / "core2_mic_imu_vibration_sound.csv"
SAVE_INTERVAL = 10
GYRO_CALIBRATION_SECONDS = 2.0

ACC_WINDOW_SIZE = 3
GYRO_WINDOW_SIZE = 3

ACC_RMS_THRESHOLD = 0.5
GYRO_RMS_THRESHOLD = 200.0

MIC_DB_THRESHOLD = 80.0
MIC_DB_REF = 1.0
MIC_DB_FLOOR = 0.0

BASELINE_ALPHA = 0.02
HISTORY_SIZE = 120
EVENT_HISTORY_SIZE = 12
LLM_COOLDOWN_SECONDS = 2.0
CAPTURE_COOLDOWN_SECONDS = 5.0
WARNING_TRIGGER_COOLDOWN_SECONDS = 5.0
IMAGE_FILE_READY_TIMEOUT_SECONDS = 2.0
IMAGE_FILE_READY_POLL_SECONDS = 0.05

CSV_COLUMNS = [
    "pc_time",
    "mic_rms",
    "acc_rms",
    "gyro_rms",
]


def rms(values) -> float:
    if not values:
        return 0.0

    return math.sqrt(sum(v * v for v in values) / len(values))


def amplitude_to_db(value: float, reference: float = MIC_DB_REF) -> float:
    if value <= 0 or reference <= 0:
        return MIC_DB_FLOOR

    return max(MIC_DB_FLOOR, 20.0 * math.log10(value / reference))


def detect_sound(mic_rms_db: float) -> bool:
    return mic_rms_db >= MIC_DB_THRESHOLD


def ensure_csv_header(csv_file: Path):
    if csv_file.exists():
        return

    with csv_file.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def save_rows(csv_file: Path, rows):
    if not rows:
        return

    with csv_file.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_COLUMNS)
        writer.writerows(rows)


class SerialPipelineRuntime:
    def __init__(
        self,
        port: str = PORT,
        baudrate: int = BAUDRATE,
        csv_file: Path = CSV_FILE,
        history_size: int = HISTORY_SIZE,
    ):
        self.port = port
        self.baudrate = baudrate
        self.csv_file = Path(csv_file)
        self.history_size = history_size

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

        self.buffer_rows = []
        self.last_save_time = time.time()
        self.last_llm_time = 0.0
        self.last_capture_time = 0.0
        self.last_warning_trigger_time = 0.0

        self.acc_dynamic_window = deque(maxlen=ACC_WINDOW_SIZE)
        self.gyro_mag_window = deque(maxlen=GYRO_WINDOW_SIZE)
        self.acc_baseline = None
        self.gyro_bias = {"x": 0.0, "y": 0.0, "z": 0.0}

        self.state = {
            "sensor_status": {
                "serial": False,
                "microphone": False,
                "imu": False,
                "camera": None,
            },
            "latest": {
                "pc_time": "",
                "mic_rms": 0.0,
                "acc_rms": 0.0,
                "gyro_rms": 0.0,
                "is_loud": False,
                "is_vibrating": False,
                "has_warning": False,
            },
            "history": {
                "mic_rms": deque(maxlen=history_size),
                "acc_rms": deque(maxlen=history_size),
                "gyro_rms": deque(maxlen=history_size),
            },
            "llm": {
                "summary": "No manual model response yet.",
                "should_capture": False,
                "updated_at": "",
            },
            "vision": {
                "summary": "Waiting for captured image.",
                "updated_at": "",
            },
            "events": deque(maxlen=EVENT_HISTORY_SIZE),
            "image_path": "",
            "last_question": "",
            "last_error": "",
        }

    def start(self):
        if self.thread and self.thread.is_alive():
            return self

        ensure_csv_header(self.csv_file)
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run_loop,
            name="serial-pipeline-runtime",
            daemon=True,
        )
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def get_snapshot(self) -> dict:
        with self.lock:
            return {
                "sensor_status": dict(self.state["sensor_status"]),
                "latest": dict(self.state["latest"]),
                "history": {
                    "mic_rms": list(self.state["history"]["mic_rms"]),
                    "acc_rms": list(self.state["history"]["acc_rms"]),
                    "gyro_rms": list(self.state["history"]["gyro_rms"]),
                },
                "llm": dict(self.state["llm"]),
                "vision": dict(self.state["vision"]),
                "events": [copy.deepcopy(event) for event in self.state["events"]],
                "image_path": self.state["image_path"],
                "last_question": self.state["last_question"],
                "last_error": self.state["last_error"],
            }

    def submit_user_question(self, question: str):
        cleaned_question = question.strip()

        if not cleaned_question:
            return

        with self.lock:
            self.state["last_question"] = cleaned_question
            payload = dict(self.state["latest"])
            self.state["llm"] = {
                "summary": "Thinking...",
                "should_capture": False,
                "updated_at": "",
            }

        if "warning_flags" not in payload:
            payload["warning_flags"] = self._build_warning_flags(payload)

        if "has_warning" not in payload:
            payload["has_warning"] = self._has_warning(payload)

        worker = threading.Thread(
            target=self._run_manual_question_worker,
            args=(cleaned_question, payload),
            name="manual-llm-worker",
            daemon=True,
        )
        worker.start()

    def run_forever(self):
        ensure_csv_header(self.csv_file)

        try:
            self._run_loop()
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            self._flush_rows(force=True)

    def calibrate_gyro(self, ser, duration: float = GYRO_CALIBRATION_SECONDS) -> dict:
        end_time = time.time() + duration
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        sample_count = 0

        while time.time() < end_time and not self.stop_event.is_set():
            raw = ser.readline()

            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "status" in data:
                continue

            imu = data.get("imu", {})
            gyro = imu.get("gyro", {})

            if not gyro:
                continue

            sum_x += float(gyro.get("x", 0.0))
            sum_y += float(gyro.get("y", 0.0))
            sum_z += float(gyro.get("z", 0.0))
            sample_count += 1

        if sample_count == 0:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        return {
            "x": sum_x / sample_count,
            "y": sum_y / sample_count,
            "z": sum_z / sample_count,
        }

    def detect_vibration(self, accel: dict, gyro: dict) -> dict:
        ax = float(accel.get("x", 0.0))
        ay = float(accel.get("y", 0.0))
        az = float(accel.get("z", 0.0))

        raw_gx = float(gyro.get("x", 0.0))
        raw_gy = float(gyro.get("y", 0.0))
        raw_gz = float(gyro.get("z", 0.0))

        gx = raw_gx - self.gyro_bias["x"]
        gy = raw_gy - self.gyro_bias["y"]
        gz = raw_gz - self.gyro_bias["z"]

        acc_mag = math.sqrt(ax * ax + ay * ay + az * az)

        if self.acc_baseline is None:
            self.acc_baseline = acc_mag
        else:
            self.acc_baseline = (
                (1.0 - BASELINE_ALPHA) * self.acc_baseline
                + BASELINE_ALPHA * acc_mag
            )

        acc_dynamic = abs(acc_mag - self.acc_baseline)
        gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)

        self.acc_dynamic_window.append(acc_dynamic)
        self.gyro_mag_window.append(gyro_mag)

        acc_rms = rms(self.acc_dynamic_window)
        gyro_rms = rms(self.gyro_mag_window)

        is_vibrating = (
            acc_rms >= ACC_RMS_THRESHOLD
            or gyro_rms >= GYRO_RMS_THRESHOLD
        )

        return {
            "acc_rms": acc_rms,
            "gyro_rms": gyro_rms,
            "is_vibrating": is_vibrating,
        }

    def _set_error(self, message: str):
        with self.lock:
            self.state["last_error"] = message

    def _clear_csv_error(self):
        with self.lock:
            current_error = self.state["last_error"]

            if (
                "Permission denied" in current_error
                or "Failed to write CSV" in current_error
                or "CSV file is locked" in current_error
            ):
                self.state["last_error"] = ""

    def _clear_transient_serial_error(self):
        with self.lock:
            current_error = self.state["last_error"]

            if (
                current_error.startswith("Non-JSON serial data:")
                or current_error.startswith("Core2 warning:")
            ):
                self.state["last_error"] = ""

    def _is_ignorable_serial_log(self, line: str) -> bool:
        return line.startswith("[WARN]") or line.startswith("[INFO]") or line.startswith("[DEBUG]")

    def _set_sensor_status(self, serial_ok: bool, mic_ok: bool, imu_ok: bool):
        with self.lock:
            self.state["sensor_status"]["serial"] = serial_ok
            self.state["sensor_status"]["microphone"] = mic_ok
            self.state["sensor_status"]["imu"] = imu_ok

    def _update_stream_state(self, payload: dict):
        with self.lock:
            self.state["latest"] = dict(payload)
            self.state["history"]["mic_rms"].append(payload["mic_rms"])
            self.state["history"]["acc_rms"].append(payload["acc_rms"])
            self.state["history"]["gyro_rms"].append(payload["gyro_rms"])

    def _set_camera_status(self, is_ok: bool):
        with self.lock:
            self.state["sensor_status"]["camera"] = is_ok

    def _get_current_image_path(self) -> str:
        with self.lock:
            return self.state["image_path"]

    def _append_event(self, event: dict):
        with self.lock:
            self.state["events"].appendleft(event)

    def _update_event(self, event_id: str, updates: dict):
        with self.lock:
            for event in self.state["events"]:
                if event.get("event_id") == event_id:
                    event.update(updates)

                    if updates.get("image_path"):
                        self.state["image_path"] = updates["image_path"]

                    break

    def _build_warning_flags(self, payload: dict) -> dict:
        return {
            "noise": payload["mic_rms"] > MIC_DB_THRESHOLD,
            "linear_acceleration": payload["acc_rms"] > ACC_RMS_THRESHOLD,
            "angular_acceleration": payload["gyro_rms"] > GYRO_RMS_THRESHOLD,
        }

    def _has_warning(self, payload: dict) -> bool:
        return any(payload["warning_flags"].values())

    def _capture_prefix(self, warning_flags: dict) -> str:
        active_flags = [name for name, active in warning_flags.items() if active]

        if not active_flags:
            return "warning"

        return "warning_" + "_".join(active_flags)

    def _wait_for_image_file(self, image_path: str, timeout_seconds: float = IMAGE_FILE_READY_TIMEOUT_SECONDS) -> str:
        if not image_path:
            return ""

        target_path = Path(image_path)
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            try:
                if target_path.exists() and target_path.stat().st_size > 0:
                    with target_path.open("rb") as file_obj:
                        if file_obj.read(1):
                            return str(target_path)
            except OSError:
                pass

            time.sleep(IMAGE_FILE_READY_POLL_SECONDS)

        return ""

    def _read_image_bytes(self, image_path: str) -> bytes:
        ready_path = self._wait_for_image_file(image_path)

        if not ready_path:
            return b""

        try:
            return Path(ready_path).read_bytes()
        except OSError:
            return b""

    def _update_llm_state(self, llm_result: dict):
        updated_at = llm_result.get(
            "updated_at",
            datetime.now().isoformat(timespec="seconds"),
        )

        with self.lock:
            self.state["llm"] = {
                "summary": llm_result.get("summary", ""),
                "should_capture": bool(llm_result.get("should_capture", False)),
                "updated_at": updated_at,
            }

    def _run_manual_question_worker(self, question: str, payload: dict):
        try:
            llm_result = analyze_sensor_payload(
                payload,
                image_path="",
                user_question=question,
                context_type="manual_query",
            )
        except Exception as exc:
            llm_result = {
                "summary": f"module3_llm failed: {exc}",
                "should_capture": False,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._set_error(llm_result["summary"])

        self._update_llm_state(llm_result)

    def _update_vision_state(self, vision_result: dict, image_path: str):
        updated_at = vision_result.get(
            "updated_at",
            datetime.now().isoformat(timespec="seconds"),
        )

        with self.lock:
            self.state["vision"] = {
                "summary": vision_result.get("summary", ""),
                "updated_at": updated_at,
            }
            self.state["image_path"] = image_path or self.state["image_path"]

    def _build_warning_event(self, payload: dict) -> dict:
        return {
            "event_id": payload.get(
                "pc_time",
                datetime.now().isoformat(timespec="milliseconds"),
            ),
            "triggered_at": payload.get(
                "pc_time",
                datetime.now().isoformat(timespec="seconds"),
            ),
            "mic_rms": payload["mic_rms"],
            "acc_rms": payload["acc_rms"],
            "gyro_rms": payload["gyro_rms"],
            "warning_flags": dict(payload["warning_flags"]),
            "llm_summary": "Thinking...",
            "llm_updated_at": "",
            "vision_summary": "",
            "image_path": "",
            "image_bytes": b"",
            "image_status": "Capturing and processing image...",
        }

    def _reset_motion_state(self):
        self.acc_dynamic_window.clear()
        self.gyro_mag_window.clear()
        self.acc_baseline = None
        self.gyro_bias = {"x": 0.0, "y": 0.0, "z": 0.0}

    def _flush_rows(self, force: bool = False):
        now_time = time.time()

        if not force and now_time - self.last_save_time < SAVE_INTERVAL:
            return

        if not self.buffer_rows:
            self.last_save_time = now_time
            return

        try:
            save_rows(self.csv_file, self.buffer_rows)
        except PermissionError as exc:
            self._set_error(f"CSV file is locked, retrying later: {exc}")
            return
        except Exception as exc:
            self._set_error(f"Failed to write CSV, retrying later: {exc}")
            return

        self.buffer_rows.clear()
        self.last_save_time = now_time
        self._clear_csv_error()

    def _run_event_llm(self, payload: dict, image_path: str = "") -> dict:
        try:
            llm_result = analyze_sensor_payload(
                payload,
                image_path=image_path,
                user_question="",
                context_type="warning_event",
            )
        except Exception as exc:
            llm_result = {
                "summary": f"module3_llm failed: {exc}",
                "should_capture": bool(payload.get("has_warning", False)),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._set_error(llm_result["summary"])

        self.last_llm_time = time.time()
        return llm_result

    def _capture_and_vision_once(self, payload: dict, now_time: float) -> tuple[str, str, dict]:
        raw_image_path = ""
        display_image_path = ""
        vision_result = {
            "summary": "No image received from module4 yet.",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "processed_image_path": "",
        }

        if not payload["has_warning"]:
            return raw_image_path, display_image_path, vision_result

        try:
            camera_result = capture_image(
                prefix=self._capture_prefix(payload["warning_flags"])
            )
            raw_image_path = self._wait_for_image_file(camera_result.get("image_path", ""))
            camera_ok = bool(camera_result.get("camera_ok", False))
            self._set_camera_status(camera_ok)

            if not camera_ok:
                self._set_error(
                    camera_result.get("message", "module4_camera did not return an image.")
                )
                return raw_image_path, display_image_path, vision_result

            self.last_capture_time = now_time
        except Exception as exc:
            self._set_camera_status(False)
            self._set_error(f"module4_camera failed: {exc}")
            return raw_image_path, display_image_path, vision_result

        if not raw_image_path:
            return raw_image_path, display_image_path, vision_result

        try:
            vision_result = analyze_image(image_path=raw_image_path, sensor_payload=payload)
            display_image_path = self._wait_for_image_file(
                vision_result.get("processed_image_path", "")
            ) or raw_image_path
            self._update_vision_state(vision_result, display_image_path)
        except Exception as exc:
            self._set_error(f"module5_vision failed: {exc}")
            vision_result = {
                "summary": f"module5_vision failed: {exc}",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "processed_image_path": "",
            }
            display_image_path = raw_image_path

        return raw_image_path, display_image_path, vision_result

    def _process_warning_event(self, event_id: str, payload: dict):
        raw_image_path, display_image_path, vision_result = self._capture_and_vision_once(
            payload,
            now_time=time.time(),
        )

        self._update_event(
            event_id,
            {
                "image_path": display_image_path,
                "image_bytes": self._read_image_bytes(display_image_path),
                "image_status": (
                    "Processed image ready."
                    if display_image_path else
                    "No image captured."
                ),
                "vision_summary": vision_result.get("summary", ""),
            },
        )

        llm_result = self._run_event_llm(payload, image_path=raw_image_path)

        self._update_event(
            event_id,
            {
                "llm_summary": llm_result.get("summary", "No LLM result returned."),
                "llm_updated_at": llm_result.get("updated_at", ""),
            },
        )

    def _maybe_handle_warning_trigger(self, payload: dict):
        if not payload["has_warning"]:
            return

        now_time = time.time()

        if now_time - self.last_warning_trigger_time < WARNING_TRIGGER_COOLDOWN_SECONDS:
            return

        self.last_warning_trigger_time = now_time
        warning_event = self._build_warning_event(payload)
        event_id = warning_event["event_id"]
        self._append_event(warning_event)

        worker = threading.Thread(
            target=self._process_warning_event,
            args=(event_id, dict(payload)),
            name=f"warning-event-{event_id}",
            daemon=True,
        )
        worker.start()

    def _run_loop(self):
        while not self.stop_event.is_set():
            try:
                with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                    self._set_error("")
                    self._set_sensor_status(True, False, False)
                    self._reset_motion_state()
                    self.gyro_bias = self.calibrate_gyro(ser)
                    self.last_save_time = time.time()

                    while not self.stop_event.is_set():
                        self._flush_rows()

                        raw = ser.readline()

                        if not raw:
                            continue

                        line = raw.decode("utf-8", errors="replace").strip()

                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            if self._is_ignorable_serial_log(line):
                                continue

                            self._set_error(f"Non-JSON serial data: {line}")
                            continue

                        self._clear_transient_serial_error()

                        if "status" in data:
                            status = str(data.get("status", "")).lower()

                            if status == "error":
                                self._set_error(str(data.get("message", "Core2 status error")))

                            continue

                        pc_time = datetime.now().isoformat(timespec="milliseconds")

                        mic = data.get("mic", {})
                        imu = data.get("imu", {})
                        accel = imu.get("accel", {})
                        gyro = imu.get("gyro", {})

                        mic_rms_raw = float(mic.get("rms", 0.0))
                        mic_rms = amplitude_to_db(mic_rms_raw)
                        is_loud = detect_sound(mic_rms)

                        mic_ok = bool(mic)
                        imu_ok = bool(accel and gyro)
                        self._set_sensor_status(True, mic_ok, imu_ok)

                        if not imu_ok:
                            payload = {
                                "pc_time": pc_time,
                                "mic_rms": mic_rms,
                                "acc_rms": 0.0,
                                "gyro_rms": 0.0,
                                "is_loud": is_loud,
                                "is_vibrating": False,
                                "warning_flags": {},
                                "has_warning": is_loud,
                            }
                            payload["warning_flags"] = self._build_warning_flags(payload)
                            payload["has_warning"] = self._has_warning(payload)
                            self._update_stream_state(payload)
                            self._maybe_handle_warning_trigger(payload)
                            self.buffer_rows.append(
                                {
                                    "pc_time": pc_time,
                                    "mic_rms": mic_rms,
                                    "acc_rms": 0.0,
                                    "gyro_rms": 0.0,
                                }
                            )
                            continue

                        vibration = self.detect_vibration(accel, gyro)
                        payload = {
                            "pc_time": pc_time,
                            "mic_rms": mic_rms,
                            "acc_rms": vibration["acc_rms"],
                            "gyro_rms": vibration["gyro_rms"],
                            "is_loud": is_loud,
                            "is_vibrating": vibration["is_vibrating"],
                        }
                        payload["warning_flags"] = self._build_warning_flags(payload)
                        payload["has_warning"] = self._has_warning(payload)

                        self._update_stream_state(payload)
                        self._maybe_handle_warning_trigger(payload)

                        self.buffer_rows.append(
                            {
                                "pc_time": pc_time,
                                "mic_rms": mic_rms,
                                "acc_rms": vibration["acc_rms"],
                                "gyro_rms": vibration["gyro_rms"],
                            }
                        )

            except serial.SerialException as exc:
                self._set_sensor_status(False, False, False)
                self._set_error(f"Failed to open serial port {self.port}: {exc}")
                time.sleep(2.0)
            except Exception as exc:
                self._set_sensor_status(False, False, False)
                self._set_error(f"Serial runtime stopped unexpectedly: {exc}")
                time.sleep(2.0)
            finally:
                self._flush_rows(force=True)


def main():
    runtime = SerialPipelineRuntime()
    runtime.run_forever()


if __name__ == "__main__":
    main()
