from datetime import datetime
from pathlib import Path

import ollama

DEFAULT_MODEL_NAME = "gemma4:e4b"


def _build_triggered_metric_text(warning_flags: dict) -> str:
    labels = {
        "noise": "mic_rms / Noise (dB)",
        "linear_acceleration": "acc_rms / Linear Acceleration",
        "angular_acceleration": "gyro_rms / Angular Acceleration",
    }

    active_metrics = [
        labels[name]
        for name, is_active in warning_flags.items()
        if is_active
    ]

    if not active_metrics:
        return "None"

    return ", ".join(active_metrics)


def build_ollama_prompt(
    sensor_payload: dict,
    trigger_time: str,
    image_path: str = "",
    user_question: str = "",
    context_type: str = "warning_event",
):
    mic_rms = float(sensor_payload.get("mic_rms", 0.0))
    acc_rms = float(sensor_payload.get("acc_rms", 0.0))
    gyro_rms = float(sensor_payload.get("gyro_rms", 0.0))
    warning_flags = sensor_payload.get("warning_flags", {})

    image_text = image_path if image_path else "No image captured."
    triggered_metrics = _build_triggered_metric_text(warning_flags)

    if context_type == "manual_query":
        return f"""
You are a motorcycle production-line quality monitoring assistant.

The operator is asking a manual question about the current live sensor snapshot.

Current snapshot:
- mic_rms / Noise (dB): {mic_rms:.2f}
- acc_rms / Linear Acceleration: {acc_rms:.4f}
- gyro_rms / Angular Acceleration: {gyro_rms:.2f}
- Current abnormal metrics: {triggered_metrics}
- Snapshot time: {trigger_time}

Operator question:
- {user_question.strip() or "Please summarize the current state."}

Please answer briefly and practically:
1. What the current data suggests.
2. Whether anything looks abnormal.
3. What the operator should check next.
""".strip()

    extra_question = ""

    if user_question.strip():
        extra_question = (
            "\nAdditional operator question:\n"
            f"- {user_question.strip()}\n"
        )

    return f"""
You are a motorcycle production quality control assistant.

The previous sensor pipeline has already detected an abnormal warning event.

Event:
- mic_rms / Noise (dB): {mic_rms:.2f}
- acc_rms / Linear Acceleration: {acc_rms:.4f}
- gyro_rms / Angular Acceleration: {gyro_rms:.2f}
- Triggered abnormal metrics: {triggered_metrics}
- Trigger time: {trigger_time}
- Captured image path: {image_text}
{extra_question}

Please inspect the situation as a production quality control assistant.

Important rule:
If the noise level is greater than 80 dB but lower than 85 dB, classify it as a warning only, not as an abnormal event.

Provide a concise result using the following format:

Detected Issue:
Abnormal vibration detected / Abnormal acoustic signal detected / Warning only, no abnormal event detected

Issue Description:
Briefly describe whether the event indicates non-compliance or a potential risk related to EU motorcycle production regulations.

Possible Causes:
List the most likely causes in no more than 30 words.

Recommended Actions:
List the most appropriate actions in no more than 30 words.
""".strip()


def call_ollama(model_name: str, prompt: str, image_path: str = "") -> str:
    message = {
        "role": "user",
        "content": prompt,
    }

    if image_path:
        message["images"] = [str(Path(image_path))]

    try:
        response = ollama.chat(
            model=model_name,
            messages=[message],
        )
        return response["message"]["content"]

    except Exception as image_error:
        if not image_path:
            raise

        fallback_prompt = (
            f"{prompt}\n\n"
            "The image was captured and displayed in the UI, but sending the "
            f"image to Ollama failed. Image-send error: {image_error}\n"
            "Please still provide a sensor-based diagnosis."
        )

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": fallback_prompt,
                }
            ],
        )

        return (
            response["message"]["content"]
            + "\n\nNote: camera image was captured and shown in the UI, "
            + f"but Ollama used text only. Image-send error: {image_error}"
        )


def analyze_sensor_payload(
    sensor_payload: dict,
    image_path: str = "",
    user_question: str = "",
    model_name: str = DEFAULT_MODEL_NAME,
    context_type: str = "warning_event",
) -> dict:
    trigger_time = sensor_payload.get(
        "pc_time",
        datetime.now().isoformat(timespec="seconds"),
    )

    prompt = build_ollama_prompt(
        sensor_payload=sensor_payload,
        trigger_time=trigger_time,
        image_path=image_path,
        user_question=user_question,
        context_type=context_type,
    )

    summary = call_ollama(
        model_name=model_name,
        prompt=prompt,
        image_path=image_path,
    )

    return {
        "summary": summary,
        "should_capture": bool(sensor_payload.get("has_warning", False)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
    }
