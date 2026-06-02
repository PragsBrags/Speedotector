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
from ingestion.zones import Zone, ZoneType
from pipeline import process_video
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


def remove_temp_video():
    video_path = st.session_state.get("video_path")
    temp_dir = st.session_state.get("video_temp_dir")

    if video_path and os.path.exists(video_path):
        os.remove(video_path)
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    for key in (
        "video_path",
        "video_temp_dir",
        "upload_signature",
        "research_zones",
        "last_zone_result",
    ):
        st.session_state.pop(key, None)


def write_uploaded_video(uploaded_file):
    remove_temp_video()
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


@st.cache_resource
def load_models(model_path):
    from ocr.licensePlate import LicensePlateDetection, PaddleInference

    return LicensePlateDetection(model_path), PaddleInference()


@st.cache_resource
def load_vehicle_detector(vehicle_model_path):
    from detection.vehicle import VehicleDetector

    return VehicleDetector(vehicle_model_path)


if not st.session_state.get("stale_temp_dirs_cleaned"):
    cleanup_stale_temp_dirs()
    st.session_state["stale_temp_dirs_cleaned"] = True

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0


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


def render_alpr_mode(video_path, first_frame_rgb, frame_width, frame_height):
    st.subheader("ALPR setup")
    use_full_frame = st.checkbox("Use full frame", value=True)
    roi = (0, 0, frame_width, frame_height)

    if not use_full_frame:
        roi = render_roi_selector(first_frame_rgb, frame_width, frame_height)
    else:
        st.image(
            first_frame_rgb,
            caption=f"Full frame preview: width={frame_width}, height={frame_height}",
            use_container_width=True,
        )

    with st.expander("Playback", expanded=False):
        st.video(video_path)

    save_to_db = st.checkbox(
        "Save results to database",
        value=False,
        help="If unchecked, detections are shown only in this session and no database connection is required.",
    )

    if st.button("Run detection", type="primary"):
        st.info("Loading models and processing video. This may take a while...")
        progress_text = st.empty()
        live_results = []

        def on_result(result):
            live_results.append(result)
            progress_text.write(f"Found {len(live_results)} detection(s)...")

        try:
            detector, ocr = load_models(model_path())
            results = process_video(
                video_path=video_path,
                roi=roi,
                save_to_db=save_to_db,
                progress_callback=on_result,
                include_images=True,
                detector=detector,
                ocr=ocr,
            )
            st.success(f"Finished. Found {len(results)} detection(s).")
            render_alpr_results(results)
        except Exception as e:
            st.error(str(e))
            st.exception(e)


def render_roi_selector(first_frame_rgb, frame_width, frame_height):
    if "roi_x" not in st.session_state:
        update_roi_state((0, 0, frame_width, frame_height))
    else:
        update_roi_state(clamp_roi(*roi_state(), frame_width, frame_height))

    control_col, canvas_col = st.columns([1, 2.2], vertical_alignment="top")
    with control_col:
        st.caption("Exact coordinates")
        x = st.number_input(
            "X",
            min_value=0,
            max_value=frame_width - 1,
            value=st.session_state["roi_x"],
        )
        y = st.number_input(
            "Y",
            min_value=0,
            max_value=frame_height - 1,
            value=st.session_state["roi_y"],
        )
        roi_w = min(st.session_state["roi_w"], frame_width - int(x))
        roi_h = min(st.session_state["roi_h"], frame_height - int(y))
        roi_w = st.number_input(
            "Width",
            min_value=1,
            max_value=frame_width - int(x),
            value=roi_w,
        )
        roi_h = st.number_input(
            "Height",
            min_value=1,
            max_value=frame_height - int(y),
            value=roi_h,
        )

        manual_roi = clamp_roi(x, y, roi_w, roi_h, frame_width, frame_height)
        if manual_roi != roi_state():
            update_roi_state(manual_roi)

    with canvas_col:
        scale, canvas_width, canvas_height, canvas_image = canvas_setup(
            first_frame_rgb, frame_width, frame_height
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
                key=f"roi_canvas_{st.session_state.get('upload_signature')}",
            )
            if canvas_result.json_data and canvas_result.json_data["objects"]:
                selected_box = canvas_result.json_data["objects"][-1]
                canvas_roi = clamp_roi(
                    selected_box["left"] / scale,
                    selected_box["top"] / scale,
                    selected_box["width"] * selected_box["scaleX"] / scale,
                    selected_box["height"] * selected_box["scaleY"] / scale,
                    frame_width,
                    frame_height,
                )
                if canvas_roi != roi_state():
                    update_roi_state(canvas_roi)
                    st.rerun()

        roi = roi_state()
        st.caption(f"ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")
        return roi


def render_alpr_results(results):
    for result in results:
        st.subheader(f"Detection {result.get('detection_id') or result['frame_index']}")
        st.write("Plate text:", result["plate_text"])
        st.write("Coordinates:", result["coords"])
        st.write(
            "Confidence:",
            f"detector {result['detector_confidence']:.2%}, "
            f"OCR {result['ocr_confidence']:.2%}",
        )

        plate_img = result["plate_img"]
        plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        st.image(plate_img_rgb, caption="Detected plate crop")


def render_zone_mode(video_path, first_frame_rgb, frame_width, frame_height):
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

    render_zone_run_panel(video_path, zones, enabled_rules, validation)


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


def render_zone_run_panel(video_path, zones, enabled_rules, validation):
    st.subheader("Run zone detection")
    settings_col, status_col = st.columns([1, 1], vertical_alignment="top")
    with settings_col:
        vehicle_model_path = st.text_input("Vehicle model", value="yolov8n.pt")
        min_vehicle_confidence = st.slider(
            "Minimum vehicle confidence", 0.0, 1.0, 0.25, 0.05
        )
        run_plate_ocr = st.checkbox("Read plate on violation evidence", value=True)
        limit_frames = st.checkbox("Limit frames for quick test", value=False)
        max_frames = None
        if limit_frames:
            max_frames = st.number_input("Maximum frames", min_value=1, value=300)

    with status_col:
        st.metric("Zones", len(zones))
        st.metric("Enabled rules", len(enabled_rules))
        st.caption("Evidence is written under `outputs/<run_id>/evidence`.")

    if st.button("Run zone violation detection", type="primary", disabled=not validation.valid):
        progress_text = st.empty()

        def on_progress(update):
            progress_text.write(
                "Frame {frame_number} | vehicles {vehicle_count} | tracks "
                "{active_tracks} | signal {signal_state} | events {events}".format(
                    **update
                )
            )

        try:
            vehicle_detector = load_vehicle_detector(vehicle_model_path)
            plate_detector = ocr = None
            if run_plate_ocr:
                plate_detector, ocr = load_models(model_path())
            result = process_zone_violations(
                video_path=video_path,
                zones=zones,
                enabled_rules=enabled_rules,
                vehicle_detector=vehicle_detector,
                plate_detector=plate_detector,
                ocr=ocr,
                progress_callback=on_progress,
                max_frames=max_frames,
                min_vehicle_confidence=min_vehicle_confidence,
                run_plate_ocr=run_plate_ocr,
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


st.title("Speedotector")
st.write("Upload a video, then choose either ALPR-only processing or zone violation detection.")

uploaded_file = st.file_uploader(
    "Upload video",
    type=["mp4", "mov", "avi", "mkv"],
    key=f"video_upload_{st.session_state['uploader_key']}",
)

if st.button("Clear uploaded video", disabled=uploaded_file is None):
    remove_temp_video()
    st.session_state["uploader_key"] += 1
    st.rerun()

if uploaded_file:
    upload_signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("upload_signature") != upload_signature:
        video_path = write_uploaded_video(uploaded_file)
    else:
        video_path = st.session_state["video_path"]

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        st.error("Could not read first frame from video.")
        st.stop()

    h, w = first_frame.shape[:2]
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    mode = st.radio(
        "Mode",
        ["ALPR only", "Zone violation detection"],
        horizontal=True,
    )

    if mode == "ALPR only":
        render_alpr_mode(video_path, first_frame_rgb, w, h)
    else:
        render_zone_mode(video_path, first_frame_rgb, w, h)
