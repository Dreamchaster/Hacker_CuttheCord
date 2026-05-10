import base64
from pathlib import Path
import time

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from module2_serial import (
    ACC_RMS_THRESHOLD,
    GYRO_RMS_THRESHOLD,
    MIC_DB_THRESHOLD,
    SerialPipelineRuntime,
)

REFRESH_INTERVAL_MS = 1000
RUNTIME_CACHE_VERSION = "runtime-v13"
IMAGE_UI_RETRY_SECONDS = 0.8
IMAGE_UI_RETRY_POLL_SECONDS = 0.1


@st.cache_resource
def get_runtime(_cache_version: str):
    return SerialPipelineRuntime()


def sensor_status_card(title: str, is_ok):
    if is_ok is None:
        dot_color = "#94a3b8"
        status_text = "Pending"
    else:
        dot_color = "#16a34a" if is_ok else "#dc2626"
        status_text = "Fine" if is_ok else "Error"

    st.markdown(
        f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 12px 16px;
            text-align: center;
            background-color: #f8f9fa;
        ">
            <div style="font-size: 16px; font-weight: 600;">{title}</div>
            <div style="margin-top: 10px; color: {dot_color}; font-weight: 600;">
                &#9679; {status_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def monitor_card(title: str, value_text: str, is_good: bool):
    status_color = "#16a34a" if is_good else "#dc2626"
    status_text = "Good" if is_good else "Warning"

    st.markdown(
        f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 14px 16px;
            background-color: #ffffff;
            min-height: 120px;
        ">
            <div style="font-size: 15px; color: #475569; margin-bottom: 12px;">{title}</div>
            <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                <div style="font-size: 32px; font-weight: 700; color: #111827;">{value_text}</div>
                <div style="font-size: 16px; font-weight: 600; color: {status_color};">
                    &#9679; {status_text}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_stream(series_name: str, values):
    if not values:
        st.info(f"Waiting for {series_name} data.")
        return

    data_frame = pd.DataFrame({series_name: values})
    st.line_chart(data_frame)


def render_event_metrics(event: dict):
    warning_flags = event.get("warning_flags", {})
    metric_rows = [
        ("Noise", float(event.get("mic_rms", 0.0)), ".2f", warning_flags.get("noise", False)),
        ("Linear Acc.", float(event.get("acc_rms", 0.0)), ".4f", warning_flags.get("linear_acceleration", False)),
        ("Angular Acc.", float(event.get("gyro_rms", 0.0)), ".2f", warning_flags.get("angular_acceleration", False)),
    ]

    st.markdown("**Abnormal Data**")
    st.caption(str(event.get("triggered_at", "")).replace("T", " "))

    for label, value, value_format, is_warning in metric_rows:
        value_color = "#dc2626" if is_warning else "#111827"
        st.markdown(
            f'{label}: <span style="color:{value_color}; font-weight:600;">{format(value, value_format)}</span>',
            unsafe_allow_html=True,
        )


def load_image_bytes(image_path: str):
    if not image_path:
        return None

    target_path = Path(image_path)
    deadline = time.time() + IMAGE_UI_RETRY_SECONDS

    while time.time() < deadline:
        try:
            if target_path.exists() and target_path.stat().st_size > 0:
                return target_path.read_bytes()
        except OSError:
            pass

        time.sleep(IMAGE_UI_RETRY_POLL_SECONDS)

    return None


def render_event_image(image_bytes, image_path: str):
    if not image_bytes:
        return False

    suffix = Path(image_path).suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    encoded_image = base64.b64encode(image_bytes).decode("ascii")

    st.markdown(
        f"""
        <img
            src="data:{mime_type};base64,{encoded_image}"
            style="
                width: 100%;
                height: auto;
                display: block;
                border-radius: 8px;
            "
        />
        """,
        unsafe_allow_html=True,
    )
    return True


st.set_page_config(
    page_title="Edge AI Enabled Motorcycle Quality Control System",
    layout="wide",
)

st.markdown(
    """
    <style>
    .st-key-ai_agent_qa_fab {
        position: fixed;
        right: 24px;
        bottom: 24px;
        z-index: 9999;
    }

    .st-key-ai_agent_qa_fab button {
        width: 58px;
        height: 58px;
        border-radius: 999px;
        padding: 0;
        font-size: 28px;
        font-weight: 700;
        background: #111827;
        color: #ffffff;
        border: 1px solid #111827;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.22);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st_autorefresh(interval=REFRESH_INTERVAL_MS, key="ui_refresh")

runtime = get_runtime(RUNTIME_CACHE_VERSION)
runtime.start()
snapshot = runtime.get_snapshot()

sensor_status = snapshot["sensor_status"]
latest = snapshot["latest"]
history = snapshot["history"]
manual_llm = snapshot["llm"]
events = snapshot.get("events", [])
last_error = snapshot["last_error"]
last_question = snapshot["last_question"]

mic_history = history.get("mic_rms", [])
acc_history = history.get("acc_rms", [])
gyro_history = history.get("gyro_rms", [])

noise_ok = latest["mic_rms"] <= MIC_DB_THRESHOLD
acc_ok = latest["acc_rms"] <= ACC_RMS_THRESHOLD
gyro_ok = latest["gyro_rms"] <= GYRO_RMS_THRESHOLD

st.title("Edge AI Enabled Motorcycle Quality Control System")

if last_error:
    st.warning(last_error)

st.subheader("Sensor Status")

sensor_cols = st.columns(3)

with sensor_cols[0]:
    sensor_status_card("Microphone", sensor_status.get("microphone"))

with sensor_cols[1]:
    sensor_status_card("IMU", sensor_status.get("imu"))

with sensor_cols[2]:
    sensor_status_card("Camera", sensor_status.get("camera"))

st.divider()

st.subheader("Real-Time Indicators")

metric_cols = st.columns(3)

with metric_cols[0]:
    monitor_card("Noise (dB)", f"{latest['mic_rms']:.2f}", noise_ok)

with metric_cols[1]:
    monitor_card("Linear Acceleration", f"{latest['acc_rms']:.4f}", acc_ok)

with metric_cols[2]:
    monitor_card("Angular Acceleration", f"{latest['gyro_rms']:.2f}", gyro_ok)

st.divider()

st.subheader("Trend Curves")

chart_cols = st.columns(3)

with chart_cols[0]:
    st.markdown("**Noise (dB)**")
    with st.container(border=True):
        plot_stream("noise_db", mic_history)

with chart_cols[1]:
    st.markdown("**Linear Acceleration**")
    with st.container(border=True):
        plot_stream("linear_acceleration", acc_history)

with chart_cols[2]:
    st.markdown("**Angular Acceleration**")
    with st.container(border=True):
        plot_stream("angular_acceleration", gyro_history)

st.divider()

with st.popover(
    "?",
    help="AI Agent Q&A",
    key="ai_agent_qa_fab",
    width="content",
):
    st.markdown("**AI Agent Q&A**")

    user_input = st.text_area(
        "Ask the model about the current live state:",
        value=last_question,
        height=180,
        placeholder="Example: Is the current noise and vibration abnormal?",
        key="user_input",
    )

    if st.button("Submit question", use_container_width=True):
        runtime.submit_user_question(user_input)
        st.success("Question sent to module3.")

    with st.container(border=True):
        st.markdown("**Model Output**")
        st.write(manual_llm["summary"])
        if manual_llm["updated_at"]:
            st.caption(f"Updated at {manual_llm['updated_at']}")

st.subheader("Triggered Events")

if not events:
    st.info("No warning event has been triggered yet.")
else:
    for event in events:
        event_cols = st.columns([1.0, 1.15, 1.35])

        with event_cols[0]:
            with st.container(border=True):
                render_event_metrics(event)

        with event_cols[1]:
            with st.container(border=True):
                st.markdown("**Edge Detection Result**")
                event_image_path = event.get("image_path", "")
                image_bytes = event.get("image_bytes") or load_image_bytes(event_image_path)
                if render_event_image(image_bytes, event_image_path):
                    pass
                else:
                    status_text = event.get("image_status") or "Capturing and processing image..."
                    st.info(status_text)

        with event_cols[2]:
            with st.container(border=True):
                st.markdown("**LLM Response**")
                st.write(event.get("llm_summary", "Thinking..."))
                if event.get("llm_updated_at"):
                    st.caption(f"Updated at {event['llm_updated_at']}")
