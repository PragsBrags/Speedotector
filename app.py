import os
import tempfile
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
        return image_utils.image_to_url(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )

    st_image.image_to_url = image_to_url

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

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


def clamp_roi(x, y, width, height, frame_width, frame_height):
    x = max(0, min(int(x), frame_width - 1))
    y = max(0, min(int(y), frame_height - 1))
    width = max(1, min(int(width), frame_width - x))
    height = max(1, min(int(height), frame_height - y))
    return x, y, width, height


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
    "Upload a video, choose an optional region of interest, and run license plate detection."
)

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

use_full_frame = st.checkbox("Use full frame", value=True)

roi = None

if uploaded_file:
    upload_signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("upload_signature") != upload_signature:
        previous_video_path = st.session_state.get("video_path")
        if previous_video_path and os.path.exists(previous_video_path):
            os.remove(previous_video_path)

        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
            temp_video.write(uploaded_file.getbuffer())
            video_path = temp_video.name

        st.session_state["upload_signature"] = upload_signature
        st.session_state["video_path"] = video_path
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

    st.subheader("Detection setup")

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
                    key=f"roi_canvas_{upload_signature}",
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

    save_to_db = st.checkbox("Save results to database", value=False)

    if st.button("Run detection"):
        st.info("Loading models and processing video. This may take a while...")

        progress_text = st.empty()

        live_results = []

        def on_result(result):
            live_results.append(result)
            progress_text.write(f"Found {len(live_results)} detection(s)...")

        try:
            results = process_video(
                video_path=video_path,
                roi=roi,
                save_to_db=save_to_db,
                progress_callback=on_result,
            )

            st.success(f"Finished. Found {len(results)} detection(s).")

            for result in results:
                st.subheader(
                    f"Detection {result.get('detection_id') or result['frame_index']}"
                )
                st.write("Plate text:", result["plate_text"])
                st.write("Coordinates:", result["coords"])

                plate_img = result["plate_img"]
                plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                st.image(plate_img_rgb, caption="Detected plate crop")

        except Exception as e:
            st.error(str(e))
