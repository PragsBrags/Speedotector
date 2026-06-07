import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st
import streamlit.elements.image as st_image
from PIL import Image
from streamlit.elements.lib import image_utils
from streamlit.elements.lib.layout_utils import LayoutConfig

if not hasattr(st_image, "image_to_url"):

    def image_to_url(
        image,
        width=None,
        clamp=True,
        channels="RGB",
        output_format="PNG",
        image_id="image",
    ):
        try:
            return image_utils.image_to_url(
                image,
                LayoutConfig(width=width),
                clamp,
                channels,
                output_format,
                image_id,
            )
        except TypeError as exc:
            raise RuntimeError(
                "streamlit-drawable-canvas is incompatible with this Streamlit "
                "version. Install the pinned versions from requirements.txt."
            ) from exc

    st_image.image_to_url = image_to_url

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

from ingestion.roi import clamp_roi
from pipeline import process_video

st.set_page_config(page_title="Speedotector", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
            max-width: 100rem;
        }

        h1 {
            margin-bottom: 0.25rem;
        }

        div[data-testid="stImage"] img,
        div[data-testid="stVideo"] video {
            max-height: 56vh;
            object-fit: contain;
        }

        div[data-testid="stFileUploader"] section {
            min-height: 4.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


TEMP_DIR_PREFIX = "speedotector_streamlit_"
STALE_TEMP_DIR_SECONDS = 24 * 60 * 60
MAX_LOG_LINES = 100
UPLOAD_STATE_KEYS = ("video_path", "video_temp_dir", "upload_signature")
DETECTION_STATE_KEYS = (
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
    "results",
    "processing_logs",
    "run_summary",
    "last_status",
)


def cleanup_stale_temp_dirs():
    temp_root = Path(tempfile.gettempdir())
    now = time.time()
    for temp_dir in temp_root.glob(f"{TEMP_DIR_PREFIX}*"):
        if not temp_dir.is_dir():
            continue
        try:
            if now - temp_dir.stat().st_mtime > STALE_TEMP_DIR_SECONDS:
                shutil.rmtree(temp_dir, ignore_errors=True)
        except OSError:
            continue


def queue_temp_cleanup(video_path=None, temp_dir=None):
    if not video_path and not temp_dir:
        return

    pending = st.session_state.setdefault("pending_temp_cleanup", [])
    cleanup_item = {"video_path": video_path, "temp_dir": temp_dir}
    if cleanup_item not in pending:
        pending.append(cleanup_item)


def cleanup_pending_temp_dirs():
    pending = st.session_state.get("pending_temp_cleanup", [])
    if not pending:
        return

    active_video_path = st.session_state.get("video_path")
    active_temp_dir = st.session_state.get("video_temp_dir")
    remaining = []

    for cleanup_item in pending:
        video_path = cleanup_item.get("video_path")
        temp_dir = cleanup_item.get("temp_dir")

        if video_path == active_video_path or temp_dir == active_temp_dir:
            remaining.append(cleanup_item)
            continue

        try:
            if temp_dir and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            elif video_path and os.path.exists(video_path):
                os.remove(video_path)
        except OSError:
            remaining.append(cleanup_item)

    if remaining:
        st.session_state["pending_temp_cleanup"] = remaining
    else:
        st.session_state.pop("pending_temp_cleanup", None)


def reset_detection_state():
    for key in DETECTION_STATE_KEYS:
        st.session_state.pop(key, None)


def reset_upload_state(queue_cleanup=True):
    video_path = st.session_state.get("video_path")
    temp_dir = st.session_state.get("video_temp_dir")

    if queue_cleanup:
        queue_temp_cleanup(video_path, temp_dir)

    for key in UPLOAD_STATE_KEYS:
        st.session_state.pop(key, None)

    reset_detection_state()
    st.session_state["uploader_key"] += 1
    st.session_state["canvas_key_version"] += 1


def write_uploaded_video(uploaded_file):
    queue_temp_cleanup(
        st.session_state.get("video_path"),
        st.session_state.get("video_temp_dir"),
    )
    reset_detection_state()
    st.session_state["canvas_key_version"] += 1

    suffix = Path(uploaded_file.name).suffix
    temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    video_path = Path(temp_dir) / f"uploaded{suffix}"
    video_path.write_bytes(uploaded_file.getbuffer())

    st.session_state["video_temp_dir"] = temp_dir
    st.session_state["video_path"] = str(video_path)
    st.session_state["upload_signature"] = (uploaded_file.name, uploaded_file.size)
    return str(video_path)


def model_path():
    return str(Path(__file__).resolve().parent / "models" / "license_plate.pt")


def vehicle_model_path():
    return "yolo26n.pt"


@st.cache_resource(show_spinner=False)
def load_vehicle_detector(vehicle_model_path):
    from ocr.vehicleDetection import VehicleDetection

    return VehicleDetection(vehicle_model_path)


@st.cache_resource(show_spinner=False)
def load_plate_detector(model_path):
    from ocr.licensePlate import LicensePlateDetection

    return LicensePlateDetection(model_path)


@st.cache_resource(show_spinner=False)
def load_ocr():
    from ocr.licensePlate import PaddleInference

    return PaddleInference()


def empty_run_summary():
    return {
        "selected_frames_processed": 0,
        "vehicles_seen": 0,
        "plates_detected": 0,
        "ocr_attempts": 0,
        "ocr_successes": 0,
        "detections_found": 0,
    }


def append_processing_log(message):
    logs = st.session_state.setdefault("processing_logs", [])
    logs.append(message)
    del logs[:-MAX_LOG_LINES]
    st.session_state["last_status"] = message


def format_percent(value):
    if value is None:
        return "-"
    return f"{value:.1%}"


def format_coords(coords):
    if coords is None:
        return "-"
    return ", ".join(str(int(value)) for value in coords)


def render_run_summary(target):
    summary = st.session_state.get("run_summary") or empty_run_summary()
    with target.container():
        cols = st.columns(6)
        cols[0].metric("Frames", summary["selected_frames_processed"])
        cols[1].metric("Vehicles", summary["vehicles_seen"])
        cols[2].metric("Plates", summary["plates_detected"])
        cols[3].metric("OCR runs", summary["ocr_attempts"])
        cols[4].metric("OCR OK", summary["ocr_successes"])
        cols[5].metric("Detections", summary["detections_found"])


def render_processing_log(target):
    logs = st.session_state.get("processing_logs", [])
    if logs:
        target.text("\n".join(logs[-MAX_LOG_LINES:]))
    else:
        target.caption("No processing logs yet.")


def render_results(results):
    if not results:
        st.info("No detections found yet.")
        return

    for result_number, result in enumerate(results, start=1):
        with st.container(border=True):
            image_col, detail_col = st.columns([1, 2.4], vertical_alignment="top")

            with image_col:
                plate_img = result.get("plate_img")
                if plate_img is not None:
                    plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                    st.image(
                        plate_img_rgb, caption="Plate crop", use_container_width=True
                    )
                else:
                    st.caption("No plate crop available.")

            with detail_col:
                st.subheader(f"{result_number}. {result['plate_text']}")
                metric_cols = st.columns(4)
                metric_cols[0].metric("Vehicle", result["vehicle_class"])
                metric_cols[1].metric(
                    "Detector", format_percent(result.get("detector_confidence"))
                )
                metric_cols[2].metric(
                    "OCR", format_percent(result.get("ocr_confidence"))
                )
                metric_cols[3].metric("Frame", result.get("source_frame_number", "-"))

                st.write(
                    f"Vehicle confidence: {format_percent(result.get('vehicle_confidence'))}"
                )
                st.write(f"Plate coordinates: `{format_coords(result.get('coords'))}`")
                st.write(
                    f"Vehicle coordinates: `{format_coords(result.get('vehicle_coords'))}`"
                )


if not st.session_state.get("stale_temp_dirs_cleaned"):
    cleanup_stale_temp_dirs()
    st.session_state["stale_temp_dirs_cleaned"] = True

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

if "canvas_key_version" not in st.session_state:
    st.session_state["canvas_key_version"] = 0

if "run_summary" not in st.session_state:
    st.session_state["run_summary"] = empty_run_summary()


def update_roi_state(roi):
    x, y, width, height = roi
    st.session_state["roi_x"] = x
    st.session_state["roi_y"] = y
    st.session_state["roi_w"] = width
    st.session_state["roi_h"] = height


def roi_state():
    return (
        st.session_state["roi_x"],
        st.session_state["roi_y"],
        st.session_state["roi_w"],
        st.session_state["roi_h"],
    )


st.title("Speedotector")
st.write(
    "Upload a video, choose an optional region of interest, and run license plate detection with live progress."
)

with st.container(border=True):
    st.subheader("Upload")
    uploaded_file = st.file_uploader(
        "Video file",
        type=["mp4", "mov", "avi", "mkv"],
        key=f"video_upload_{st.session_state['uploader_key']}",
    )

if uploaded_file is None and st.session_state.get("video_path"):
    reset_upload_state(queue_cleanup=True)
    st.rerun()

video_path = None
upload_signature = None
roi = None

if uploaded_file:
    upload_signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("upload_signature") != upload_signature:
        video_path = write_uploaded_video(uploaded_file)
    else:
        video_path = st.session_state["video_path"]

has_video = video_path is not None

with st.sidebar:
    st.header("Settings")
    use_full_frame = st.checkbox("Use full frame", value=True)
    save_to_db = st.checkbox(
        "Save results to database",
        value=False,
        help="If unchecked, detections are shown only in this session and no database connection is required.",
    )
    run_detection = st.button("Run detection", disabled=not has_video, type="primary")
    clear_upload = st.button("Clear uploaded video", disabled=not has_video)

if clear_upload:
    reset_upload_state(queue_cleanup=True)
    st.rerun()

if has_video:
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        st.error("Could not read first frame from video.")
        st.stop()

    h, w = first_frame.shape[:2]
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    st.subheader("ROI preview and selection")

    if use_full_frame:
        roi = (0, 0, w, h)
    else:
        if "roi_x" not in st.session_state:
            update_roi_state((0, 0, w, h))
        else:
            update_roi_state(clamp_roi(*roi_state(), w, h))

        control_col, canvas_col = st.columns([1, 2.2], vertical_alignment="top")

        with control_col:
            st.caption("Exact coordinates")
            x = st.number_input(
                "X",
                min_value=0,
                max_value=w - 1,
                value=st.session_state["roi_x"],
            )
            y = st.number_input(
                "Y",
                min_value=0,
                max_value=h - 1,
                value=st.session_state["roi_y"],
            )
            roi_w = min(st.session_state["roi_w"], w - int(x))
            roi_h = min(st.session_state["roi_h"], h - int(y))
            roi_w = st.number_input(
                "Width",
                min_value=1,
                max_value=w - int(x),
                value=roi_w,
            )
            roi_h = st.number_input(
                "Height",
                min_value=1,
                max_value=h - int(y),
                value=roi_h,
            )

            manual_roi = clamp_roi(x, y, roi_w, roi_h, w, h)
            if manual_roi != roi_state():
                update_roi_state(manual_roi)

        with canvas_col:
            max_canvas_width = 980
            max_canvas_height = 520
            scale = min(max_canvas_width / w, max_canvas_height / h, 1)
            canvas_width = int(w * scale)
            canvas_height = int(h * scale)
            canvas_image = Image.fromarray(first_frame_rgb).resize(
                (canvas_width, canvas_height)
            )
            current_roi = roi_state()

            if st_canvas is None:
                st.error(
                    "Install streamlit-drawable-canvas to draw ROI boxes: pip install streamlit-drawable-canvas"
                )
                st.image(canvas_image, caption="ROI selector unavailable")
            else:
                initial_drawing = {
                    "version": "4.4.0",
                    "objects": [
                        {
                            "type": "rect",
                            "left": current_roi[0] * scale,
                            "top": current_roi[1] * scale,
                            "width": current_roi[2] * scale,
                            "height": current_roi[3] * scale,
                            "fill": "rgba(255, 75, 75, 0.18)",
                            "stroke": "#ff4b4b",
                            "strokeWidth": 3,
                        }
                    ],
                }

                canvas_result = st_canvas(
                    fill_color="rgba(255, 75, 75, 0.18)",
                    stroke_width=3,
                    stroke_color="#ff4b4b",
                    background_image=canvas_image,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect",
                    initial_drawing=initial_drawing,
                    key=f"roi_canvas_{st.session_state['canvas_key_version']}",
                )

                if canvas_result.json_data and canvas_result.json_data["objects"]:
                    selected_box = canvas_result.json_data["objects"][-1]
                    canvas_roi = clamp_roi(
                        selected_box["left"] / scale,
                        selected_box["top"] / scale,
                        selected_box["width"] * selected_box["scaleX"] / scale,
                        selected_box["height"] * selected_box["scaleY"] / scale,
                        w,
                        h,
                    )
                    if canvas_roi != roi_state():
                        update_roi_state(canvas_roi)
                        st.rerun()

            roi = roi_state()
            st.caption(f"ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")

    if use_full_frame:
        st.image(
            first_frame_rgb,
            caption=f"Full frame preview: width={w}, height={h}",
            use_container_width=True,
        )

    with st.expander("Playback", expanded=False):
        st.video(video_path)

    st.subheader("Processing status")
    status_placeholder = st.empty()
    summary_placeholder = st.empty()
    with st.expander("Detailed processing log", expanded=False):
        log_placeholder = st.empty()

    if st.session_state.get("last_status"):
        status_placeholder.info(st.session_state["last_status"])
    else:
        status_placeholder.caption("Ready to run detection.")
    render_run_summary(summary_placeholder)
    render_processing_log(log_placeholder)

    if run_detection:
        st.session_state["results"] = []
        st.session_state["processing_logs"] = []
        st.session_state["run_summary"] = empty_run_summary()
        st.session_state["last_status"] = "Loading models..."
        status_placeholder.info("Loading models...")
        render_run_summary(summary_placeholder)
        render_processing_log(log_placeholder)

        def record_status(message):
            print(message, flush=True)
            append_processing_log(message)
            status_placeholder.info(message)
            render_processing_log(log_placeholder)

        record_status("Loading vehicle detector...")
        live_results = []

        def on_result(result):
            live_results.append(result)

        def on_status(payload):
            message = payload.get("message", "Processing video...")
            event = payload.get("event")
            summary = st.session_state.setdefault("run_summary", empty_run_summary())

            if event == "frame_started":
                summary["selected_frames_processed"] = max(
                    summary["selected_frames_processed"],
                    int(payload.get("frame_index", 0)) + 1,
                )
            elif event == "vehicles_detected":
                summary["vehicles_seen"] += int(payload.get("vehicle_count", 0))
            elif event == "plate_detected":
                summary["plates_detected"] += 1
            elif event == "ocr_started":
                summary["ocr_attempts"] += 1
            elif event == "detection_succeeded":
                summary["ocr_successes"] += 1
                summary["detections_found"] = int(
                    payload.get("detection_count", summary["detections_found"])
                )

            append_processing_log(message)
            status_placeholder.info(message)
            render_run_summary(summary_placeholder)
            render_processing_log(log_placeholder)

        try:
            with st.status("Loading models...", expanded=True) as status:
                st.write("Loading vehicle detector...")
                vehicle_detector = load_vehicle_detector(vehicle_model_path())

                record_status("Loading license plate detector...")
                st.write("Loading license plate detector...")
                plate_detector = load_plate_detector(model_path())

                record_status("Loading OCR model...")
                st.write("Loading OCR model...")
                ocr = load_ocr()

                record_status("Models loaded")
                record_status("Scanning video frames...")
                status.update(label="Models loaded", state="complete")

            results = process_video(
                video_path=video_path,
                roi=roi,
                save_to_db=save_to_db,
                progress_callback=on_result,
                include_images=True,
                vehicle_detector=vehicle_detector,
                plate_detector=plate_detector,
                ocr=ocr,
                status_callback=on_status,
            )

            st.session_state["results"] = results
            st.session_state["run_summary"]["detections_found"] = len(results)
            st.session_state["last_status"] = (
                f"Finished. Found {len(results)} detection(s)."
            )
            st.success(f"Finished. Found {len(results)} detection(s).")
            status_placeholder.success(st.session_state["last_status"])
            render_run_summary(summary_placeholder)

        except Exception as e:
            st.error(str(e))
            st.exception(e)

    st.subheader("Results")
    render_results(st.session_state.get("results", []))
else:
    st.info("Upload a video to configure ROI and run detection.")

cleanup_pending_temp_dirs()
