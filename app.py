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

import pipeline
from ingestion.roi import clamp_roi
from ingestion.zones import Zone, ZoneType
from ui_zones import (
    RULE_OPTIONS,
    ZONE_TYPES,
    canvas_object_to_zone,
    validate_zone_setup,
    zone_config_json,
    zone_records,
    zones_to_canvas_objects,
)
from violation_pipeline import process_zone_violations

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "speedotector_matplotlib"),
)


def image_to_url(
    image,
    width=None,
    clamp=True,
    channels="RGB",
    output_format="PNG",
    image_id="image",
):
    layout_config = (
        width if isinstance(width, LayoutConfig) else LayoutConfig(width=width)
    )
    return image_utils.image_to_url(
        image,
        layout_config,
        clamp,
        channels,
        output_format,
        image_id,
    )


st_image.image_to_url = image_to_url

import streamlit_image_annotation.Detection as detection_module  # noqa: E402

detection_module.image_to_url = image_to_url
detection = detection_module.detection

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

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
ROI_LABEL = "ROI"
ROI_SELECTOR_MAX_WIDTH = 760
ROI_SELECTOR_MAX_HEIGHT = 560
UPLOAD_STATE_KEYS = ("video_path", "video_temp_dir", "upload_signature")
DETECTION_STATE_KEYS = (
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
    "roi_selected",
    "results",
    "research_zones",
    "processing_logs",
    "run_summary",
    "last_zone_result",
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


def write_uploaded_video(uploaded_file):
    queue_temp_cleanup(
        st.session_state.get("video_path"),
        st.session_state.get("video_temp_dir"),
    )
    reset_detection_state()

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


def database_configured():
    return bool(
        os.getenv("DATABASE_URL")
        or os.getenv("DB_HOST")
        or os.getenv("DB_USER")
        or os.getenv("DB_PASSWORD")
        or os.getenv("DB_NAME")
    )


@st.cache_resource(show_spinner=False)
def load_vehicle_detector(vehicle_model_path):
    from ocr.vehicleDetection import VehicleDetection

    return VehicleDetection(vehicle_model_path)


@st.cache_resource(show_spinner=False)
def load_zone_vehicle_detector(vehicle_model_path):
    from detection.vehicle import VehicleDetector

    return VehicleDetector(vehicle_model_path)


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
        "raw_frames_scanned": 0,
        "raw_frames_total": 0,
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


def default_roi(frame_width, frame_height):
    width = max(1, int(frame_width * 0.45))
    height = max(1, int(frame_height * 0.35))
    x = max(0, int((frame_width - width) / 2))
    y = max(0, int((frame_height - height) / 2))
    return (x, y, width, height)


def roi_area_percent(roi, frame_width, frame_height):
    _, _, roi_width, roi_height = roi
    frame_area = max(1, frame_width * frame_height)
    return (roi_width * roi_height / frame_area) * 100


def roi_crop(frame_rgb, roi):
    x, y, width, height = roi
    return frame_rgb[y : y + height, x : x + width]


def render_run_summary(target):
    summary = st.session_state.get("run_summary") or empty_run_summary()
    with target.container():
        cols = st.columns(7)
        raw_total = summary.get("raw_frames_total", 0)
        raw_value = summary.get("raw_frames_scanned", 0)
        raw_label = f"{raw_value}/{raw_total}" if raw_total else raw_value
        cols[0].metric("Raw frames", raw_label)
        cols[1].metric("Selected frames", summary["selected_frames_processed"])
        cols[2].metric("Vehicles", summary["vehicles_seen"])
        cols[3].metric("Plates", summary["plates_detected"])
        cols[4].metric("OCR runs", summary["ocr_attempts"])
        cols[5].metric("OCR OK", summary["ocr_successes"])
        cols[6].metric("Detections", summary["detections_found"])


def render_scan_progress(target):
    summary = st.session_state.get("run_summary") or empty_run_summary()
    raw_frames_scanned = int(summary.get("raw_frames_scanned", 0))
    raw_frames_total = int(summary.get("raw_frames_total", 0))

    if raw_frames_total > 0:
        progress = min(raw_frames_scanned / raw_frames_total, 1.0)
        target.progress(
            progress,
            text=(
                "Scanning video frames: "
                f"{raw_frames_scanned} / {raw_frames_total} raw frames"
            ),
        )
    elif raw_frames_scanned > 0:
        target.progress(
            0,
            text=f"Scanning video frames: {raw_frames_scanned} raw frames",
        )
    else:
        target.empty()


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


def parse_signal_intervals(raw_value):
    intervals = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        start_text, separator, end_text = part.partition("-")
        if not separator:
            raise ValueError("Signal intervals must use start-end pairs.")
        start = int(start_text.strip())
        end = int(end_text.strip())
        if end < start:
            raise ValueError("Signal interval end frame must be after start frame.")
        intervals.append((start, end))
    return intervals


def render_zone_mode(video_path, first_frame_rgb, frame_width, frame_height, save_to_db):
    st.subheader("Zone violation setup")
    st.session_state.setdefault("research_zones", [])
    zones = st.session_state["research_zones"]

    setup_col, canvas_col = st.columns([1.05, 2.2], vertical_alignment="top")
    with setup_col:
        enabled_rules = st.multiselect(
            "Violation types",
            options=list(RULE_OPTIONS),
            default=["restricted_zone_violation"],
            format_func=lambda value: RULE_OPTIONS[value],
        )
        zone_type = st.selectbox(
            "Zone to add",
            options=list(ZONE_TYPES),
            format_func=lambda value: ZONE_TYPES[value]["label"],
        )
        zone_style = ZONE_TYPES[zone_type]
        same_type_count = sum(zone.type == zone_type for zone in zones) + 1
        zone_id = st.text_input(
            "Zone ID",
            value=f"{zone_type}_{same_type_count}",
            key=f"zone_id_{zone_type}_{len(zones)}",
        )
        zone_label = st.text_input(
            "Label",
            value=zone_style["label"],
            key=f"zone_label_{zone_type}_{len(zones)}",
        )

        st.caption("Exact coordinates")
        if zone_type == ZoneType.STOP_LINE.value:
            manual_zone = manual_line_zone(
                zone_id, zone_type, zone_label, frame_width, frame_height
            )
        else:
            manual_zone = manual_rect_zone(
                zone_id, zone_type, zone_label, frame_width, frame_height
            )
        if st.button("Add from coordinates", use_container_width=True):
            zones.append(manual_zone)
            st.session_state["research_zones"] = zones
            st.rerun()

        validation = validate_zone_setup(zones, enabled_rules)
        if validation.valid:
            st.success("Zone setup is ready.")
        else:
            for message in validation.messages:
                st.warning(message)

        if zones:
            delete_zone_id = st.selectbox(
                "Remove zone",
                options=[zone.id for zone in zones],
                index=0,
            )
            if st.button("Remove selected zone", use_container_width=True):
                st.session_state["research_zones"] = [
                    zone for zone in zones if zone.id != delete_zone_id
                ]
                st.rerun()

        config_json = zone_config_json(video_path, zones, enabled_rules, 10, 10)
        st.download_button(
            "Download zone config",
            data=config_json,
            file_name="speedotector_zones.json",
            mime="application/json",
            use_container_width=True,
        )

    with canvas_col:
        scale, canvas_width, canvas_height, canvas_image = canvas_setup(
            first_frame_rgb, frame_width, frame_height
        )
        if st_canvas is None:
            st.error(
                "Install streamlit-drawable-canvas to draw zones, or use exact coordinates."
            )
            st.image(canvas_image, caption="Zone canvas unavailable")
            canvas_result = None
        else:
            canvas_result = st_canvas(
                fill_color=zone_style["fill"],
                stroke_width=4 if zone_type == ZoneType.STOP_LINE.value else 3,
                stroke_color=zone_style["color"],
                background_image=canvas_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=zone_style["drawing_mode"],
                initial_drawing={
                    "version": "4.4.0",
                    "objects": zones_to_canvas_objects(zones, scale),
                },
                key=(
                    f"zone_canvas_{st.session_state.get('upload_signature')}_"
                    f"{zone_type}_{len(zones)}"
                ),
            )

        add_col, clear_col = st.columns(2)
        with add_col:
            if st.button("Add drawn zone", type="secondary", use_container_width=True):
                add_drawn_zone(
                    canvas_result,
                    zones,
                    zone_type,
                    zone_id,
                    zone_label,
                    scale,
                    frame_width,
                    frame_height,
                )
        with clear_col:
            if st.button("Clear all zones", use_container_width=True):
                st.session_state["research_zones"] = []
                st.rerun()

    if zones:
        st.dataframe(zone_records(zones), use_container_width=True, hide_index=True)

    render_zone_run_panel(video_path, zones, enabled_rules, validation, save_to_db)


def manual_rect_zone(zone_id, zone_type, zone_label, frame_width, frame_height):
    x = st.number_input("Zone X", 0, frame_width - 1, 0)
    y = st.number_input("Zone Y", 0, frame_height - 1, 0)
    width = st.number_input("Zone width", 1, frame_width - int(x), min(200, frame_width))
    height = st.number_input(
        "Zone height", 1, frame_height - int(y), min(120, frame_height)
    )
    x, y, width, height = clamp_roi(x, y, width, height, frame_width, frame_height)
    points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    return Zone(id=zone_id, type=zone_type, points=points, label=zone_label)


def manual_line_zone(zone_id, zone_type, zone_label, frame_width, frame_height):
    x1 = st.number_input("Line X1", 0, frame_width - 1, 0)
    y1 = st.number_input("Line Y1", 0, frame_height - 1, frame_height // 2)
    x2 = st.number_input("Line X2", 0, frame_width - 1, frame_width - 1)
    y2 = st.number_input("Line Y2", 0, frame_height - 1, frame_height // 2)
    start = clamp_roi(x1, y1, 1, 1, frame_width, frame_height)[:2]
    end = clamp_roi(x2, y2, 1, 1, frame_width, frame_height)[:2]
    return Zone(id=zone_id, type=zone_type, points=[start, end], label=zone_label)


def add_drawn_zone(
    canvas_result,
    zones,
    zone_type,
    zone_id,
    zone_label,
    scale,
    frame_width,
    frame_height,
):
    if canvas_result is None or not canvas_result.json_data:
        st.warning("Draw a zone on the preview first.")
        return

    objects = canvas_result.json_data.get("objects", [])
    if len(objects) <= len(zones):
        st.warning("Draw a new zone before adding it.")
        return

    zone = canvas_object_to_zone(
        objects[-1],
        zone_type,
        zone_id,
        zone_label,
        scale,
        frame_width,
        frame_height,
    )
    zones.append(zone)
    st.session_state["research_zones"] = zones
    st.rerun()


def render_zone_run_panel(video_path, zones, enabled_rules, validation, save_to_db):
    st.subheader("Run zone detection")
    settings_col, status_col = st.columns([1, 1], vertical_alignment="top")
    with settings_col:
        zone_vehicle_model_path = st.text_input("Vehicle model", value="yolov8n.pt")
        min_vehicle_confidence = st.slider(
            "Minimum vehicle confidence", 0.0, 1.0, 0.25, 0.05
        )
        run_plate_ocr = st.checkbox("Read plate on violation evidence", value=True)
        manual_red_intervals_raw = st.text_input(
            "Manual prohibited intervals",
            value="",
            placeholder="120-180, 260-320",
            help="Optional frame ranges used as red/prohibited signal intervals.",
        )
        limit_frames = st.checkbox("Limit frames for quick test", value=False)
        max_frames = None
        if limit_frames:
            max_frames = st.number_input("Maximum frames", min_value=1, value=300)

    with status_col:
        st.metric("Zones", len(zones))
        st.metric("Enabled rules", len(enabled_rules))
        st.caption("Evidence is written under `outputs/<run_id>/evidence`.")
        st.caption(
            "Database persistence follows the sidebar setting and is disabled unless configured."
        )

    if st.button(
        "Run zone violation detection",
        type="primary",
        disabled=not validation.valid,
    ):
        progress_text = st.empty()

        def on_progress(update):
            progress_text.write(
                "Frame {frame_number} | vehicles {vehicle_count} | tracks "
                "{active_tracks} | signal {signal_state} | events {events}".format(
                    **update
                )
            )

        try:
            manual_signal_intervals = parse_signal_intervals(manual_red_intervals_raw)
            zone_vehicle_detector = load_zone_vehicle_detector(zone_vehicle_model_path)
            plate_detector = ocr = None
            if run_plate_ocr:
                plate_detector = load_plate_detector(model_path())
                ocr = load_ocr()
            result = process_zone_violations(
                video_path=video_path,
                zones=zones,
                enabled_rules=enabled_rules,
                vehicle_detector=zone_vehicle_detector,
                plate_detector=plate_detector,
                ocr=ocr,
                progress_callback=on_progress,
                max_frames=max_frames,
                min_vehicle_confidence=min_vehicle_confidence,
                run_plate_ocr=run_plate_ocr,
                save_to_db=save_to_db,
                manual_signal_intervals=manual_signal_intervals,
            )
            st.session_state["last_zone_result"] = result
            st.success(f"Finished. Found {len(result['events'])} violation event(s).")
        except Exception as e:
            st.error(str(e))
            st.exception(e)

    render_zone_results()


def render_zone_results():
    result = st.session_state.get("last_zone_result")
    if not result:
        return

    st.subheader("Evidence records")
    st.write(f"Processed frames: {result['processed_frames']}")
    results_path = Path(result["results_path"])
    if results_path.exists():
        st.download_button(
            "Download results JSON",
            data=results_path.read_text(encoding="utf-8"),
            file_name=results_path.name,
            mime="application/json",
        )

    for index, event in enumerate(result["events"], start=1):
        with st.expander(
            f"{index}. {event['violation_type']} | frame {event['frame_number']}",
            expanded=index == 1,
        ):
            st.write("Reason:", event["reason"])
            st.write("Track:", event["track_id"])
            st.write("Zone:", event["zone_id"])
            st.write("Evidence frame:", event.get("evidence_frame_number", "-"))
            st.write("Plate:", event.get("plate_text") or "No plate text")
            evidence_path = event.get("evidence_frame_path")
            if evidence_path and Path(evidence_path).exists():
                st.image(evidence_path, caption="Evidence frame", use_container_width=True)
            plate_path = event.get("plate_crop_path")
            if plate_path and Path(plate_path).exists():
                st.image(plate_path, caption="Plate crop")


def canvas_setup(first_frame_rgb, frame_width, frame_height):
    max_canvas_width = 980
    max_canvas_height = 520
    scale = min(max_canvas_width / frame_width, max_canvas_height / frame_height, 1)
    canvas_width = int(frame_width * scale)
    canvas_height = int(frame_height * scale)
    canvas_image = Image.fromarray(first_frame_rgb).resize((canvas_width, canvas_height))
    return scale, canvas_width, canvas_height, canvas_image


if not st.session_state.get("stale_temp_dirs_cleaned"):
    cleanup_stale_temp_dirs()
    st.session_state["stale_temp_dirs_cleaned"] = True

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

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
    "Upload a video, then choose either ALPR-only processing or zone violation detection."
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
    has_database_config = database_configured()
    mode = st.radio(
        "Mode",
        ["ALPR only", "Zone violation detection"],
        horizontal=False,
    )
    if mode == "ALPR only":
        use_full_frame = st.checkbox(
            "Use full frame",
            value=False,
            help="CLI mode asks you to select an ROI. Leave this off to match that behavior.",
        )
    else:
        use_full_frame = True
    save_to_db = st.checkbox(
        "Save results to database",
        value=False,
        disabled=not has_database_config,
        help=(
            "Configure DATABASE_URL or DB_* environment variables to enable persistence."
            if not has_database_config
            else "If unchecked, detections are shown only in this session and no database connection is required."
        ),
    )
    run_detection = (
        st.button("Run detection", disabled=not has_video, type="primary")
        if mode == "ALPR only"
        else False
    )
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

    if mode == "Zone violation detection":
        render_zone_mode(video_path, first_frame_rgb, w, h, save_to_db)
        cleanup_pending_temp_dirs()
        st.stop()

    st.subheader("ROI preview and selection")

    if use_full_frame:
        roi = (0, 0, w, h)
        st.session_state["roi_selected"] = True
    else:
        if "roi_x" not in st.session_state:
            update_roi_state(default_roi(w, h))
            st.session_state["roi_selected"] = False
        else:
            update_roi_state(clamp_roi(*roi_state(), w, h))

        st.caption("Exact coordinates")
        x_col, y_col, width_col, height_col, action_col, status_col = st.columns(
            [1, 1, 1, 1, 0.8, 1.4],
            vertical_alignment="bottom",
        )

        with x_col:
            x = st.number_input(
                "X",
                min_value=0,
                max_value=w - 1,
                value=st.session_state["roi_x"],
            )

        with y_col:
            y = st.number_input(
                "Y",
                min_value=0,
                max_value=h - 1,
                value=st.session_state["roi_y"],
            )

        roi_w = min(st.session_state["roi_w"], w - int(x))
        roi_h = min(st.session_state["roi_h"], h - int(y))

        with width_col:
            roi_w = st.number_input(
                "Width",
                min_value=1,
                max_value=w - int(x),
                value=roi_w,
            )

        with height_col:
            roi_h = st.number_input(
                "Height",
                min_value=1,
                max_value=h - int(y),
                value=roi_h,
            )

        manual_roi = clamp_roi(x, y, roi_w, roi_h, w, h)
        if manual_roi != roi_state():
            update_roi_state(manual_roi)
            st.session_state["roi_selected"] = True

        with action_col:
            if st.button("Confirm ROI"):
                st.session_state["roi_selected"] = True

        current_area = roi_area_percent(roi_state(), w, h)
        with status_col:
            if st.session_state.get("roi_selected"):
                st.success(f"Selected ({current_area:.1f}% of frame).")
            else:
                st.warning("Select or confirm ROI.")

        roi = roi_state()
        selector_col, crop_col = st.columns([2.1, 1], vertical_alignment="top")

        with selector_col:
            annotated_boxes = detection(
                image_path=first_frame_rgb,
                label_list=[ROI_LABEL],
                bboxes=[list(roi)],
                labels=[0],
                width=ROI_SELECTOR_MAX_WIDTH,
                height=ROI_SELECTOR_MAX_HEIGHT,
                line_width=3,
                use_space=True,
                key=f"roi_annotation_{upload_signature[0]}_{upload_signature[1]}",
            )

            if isinstance(annotated_boxes, list) and annotated_boxes:
                latest_box = annotated_boxes[-1]["bbox"]
                annotation_roi = clamp_roi(
                    latest_box[0],
                    latest_box[1],
                    latest_box[2],
                    latest_box[3],
                    w,
                    h,
                )
                if annotation_roi != roi_state():
                    update_roi_state(annotation_roi)
                    st.session_state["roi_selected"] = True
                    roi = annotation_roi

            st.caption(f"ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")

        with crop_col:
            st.image(
                roi_crop(first_frame_rgb, roi),
                caption="Selected ROI crop",
                use_container_width=True,
            )

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
    scan_progress_placeholder = st.empty()
    summary_placeholder = st.empty()
    with st.expander("Detailed processing log", expanded=False):
        log_placeholder = st.empty()

    if st.session_state.get("last_status"):
        status_placeholder.info(st.session_state["last_status"])
    else:
        status_placeholder.caption("Ready to run detection.")
    render_scan_progress(scan_progress_placeholder)
    render_run_summary(summary_placeholder)
    render_processing_log(log_placeholder)

    if run_detection and not st.session_state.get("roi_selected"):
        st.error("Select or confirm an ROI first. This matches the CLI ROI step.")

    if run_detection and st.session_state.get("roi_selected"):
        st.session_state["results"] = []
        st.session_state["processing_logs"] = []
        st.session_state["run_summary"] = empty_run_summary()
        st.session_state["last_status"] = "Loading models..."
        st.session_state["last_ui_status_update"] = 0.0
        status_placeholder.info("Loading models...")
        render_scan_progress(scan_progress_placeholder)
        render_run_summary(summary_placeholder)
        render_processing_log(log_placeholder)

        def refresh_processing_ui(force=False):
            now = time.monotonic()
            last_update = st.session_state.get("last_ui_status_update", 0.0)
            if not force and now - last_update < 0.5:
                return

            st.session_state["last_ui_status_update"] = now
            status_placeholder.info(
                st.session_state.get("last_status", "Processing...")
            )
            render_scan_progress(scan_progress_placeholder)
            render_run_summary(summary_placeholder)
            render_processing_log(log_placeholder)

        def record_status(message, force=False):
            print(message, flush=True)
            append_processing_log(message)
            refresh_processing_ui(force=force)

        def record_raw_scan_progress(frames_scanned, total_frames, force=False):
            summary = st.session_state.setdefault("run_summary", empty_run_summary())
            summary["raw_frames_scanned"] = max(
                int(summary.get("raw_frames_scanned", 0)),
                int(frames_scanned),
            )
            summary["raw_frames_total"] = max(
                int(summary.get("raw_frames_total", 0)),
                int(total_frames or 0),
            )
            refresh_processing_ui(force=force)

        def wrap_selected_frames_with_raw_progress(original_selected_frames):
            def selected_frames_with_raw_progress(cap, roi=None):
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

                class ProgressCapture:
                    def __init__(self, wrapped_cap):
                        self._wrapped_cap = wrapped_cap
                        self.frames_scanned = 0

                    def read(self):
                        ret, frame = self._wrapped_cap.read()
                        if ret:
                            self.frames_scanned += 1
                            if (
                                self.frames_scanned == 1
                                or self.frames_scanned % 15 == 0
                            ):
                                record_raw_scan_progress(
                                    self.frames_scanned,
                                    total_frames,
                                )
                        else:
                            record_raw_scan_progress(
                                self.frames_scanned,
                                total_frames,
                                force=True,
                            )
                        return ret, frame

                    def __getattr__(self, name):
                        return getattr(self._wrapped_cap, name)

                progress_cap = ProgressCapture(cap)
                try:
                    yield from original_selected_frames(progress_cap, roi=roi)
                finally:
                    record_raw_scan_progress(
                        progress_cap.frames_scanned,
                        total_frames,
                        force=True,
                    )

            return selected_frames_with_raw_progress

        record_status("Loading vehicle detector...", force=True)
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
            refresh_processing_ui(force=event in {"detection_succeeded", "ocr_failed"})

        try:
            with st.status("Loading models...", expanded=True) as status:
                st.write("Loading vehicle detector...")
                vehicle_detector = load_vehicle_detector(vehicle_model_path())

                record_status("Loading license plate detector...", force=True)
                st.write("Loading license plate detector...")
                plate_detector = load_plate_detector(model_path())

                record_status("Loading OCR model...", force=True)
                st.write("Loading OCR model...")
                ocr = load_ocr()

                record_status("Models loaded", force=True)
                record_status("Preparing video scan...", force=True)
                status.update(label="Models loaded", state="complete")

            original_selected_frames = pipeline.selected_frames
            pipeline.selected_frames = wrap_selected_frames_with_raw_progress(
                original_selected_frames
            )
            try:
                results = pipeline.process_video(
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
            finally:
                pipeline.selected_frames = original_selected_frames

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
