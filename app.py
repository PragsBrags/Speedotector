import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

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
            padding-top: 3rem;
            padding-bottom: 1rem;
            max-width: 100rem;
        }

        h1 {
            margin-bottom: 0.25rem;
        }

        div[data-testid="stImage"] img,
        div[data-testid="stVideo"] video {
            max-height: 58vh;
            object-fit: contain;
        }

        div[data-testid="stFileUploader"] section {
            min-height: 4.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
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

    with st.expander("Video preview", expanded=False):
        st.video(video_path)

    st.subheader("Region of interest")

    if use_full_frame:
        roi = (0, 0, w, h)
        st.info(f"Using full frame: x=0, y=0, width={w}, height={h}")
        st.image(
            first_frame_rgb, caption="First frame preview", use_container_width=True
        )
    else:
        if "roi_x" not in st.session_state:
            st.session_state["roi_x"] = 0
            st.session_state["roi_y"] = 0
            st.session_state["roi_w"] = w
            st.session_state["roi_h"] = h
        else:
            st.session_state["roi_x"] = min(int(st.session_state["roi_x"]), w - 1)
            st.session_state["roi_y"] = min(int(st.session_state["roi_y"]), h - 1)
            st.session_state["roi_w"] = min(
                int(st.session_state["roi_w"]), w - st.session_state["roi_x"]
            )
            st.session_state["roi_h"] = min(
                int(st.session_state["roi_h"]), h - st.session_state["roi_y"]
            )

        selector_col, input_col = st.columns([3, 1])

        with selector_col:
            if st_canvas is None:
                st.warning(
                    "Install streamlit-drawable-canvas to draw ROI boxes. Manual inputs are still available."
                )
                st.image(
                    first_frame_rgb,
                    caption="First frame preview",
                    use_container_width=True,
                )
            else:
                max_canvas_width = 960
                max_canvas_height = 520
                scale = min(max_canvas_width / w, max_canvas_height / h, 1)
                canvas_width = int(w * scale)
                canvas_height = int(h * scale)
                canvas_image = Image.fromarray(first_frame_rgb).resize(
                    (canvas_width, canvas_height)
                )

                canvas_result = st_canvas(
                    fill_color="rgba(255, 75, 75, 0.18)",
                    stroke_width=3,
                    stroke_color="#ff4b4b",
                    background_image=canvas_image,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect",
                    key="roi_canvas",
                )

                if canvas_result.json_data and canvas_result.json_data["objects"]:
                    selected_box = canvas_result.json_data["objects"][-1]
                    st.session_state["roi_x"] = int(selected_box["left"] / scale)
                    st.session_state["roi_y"] = int(selected_box["top"] / scale)
                    st.session_state["roi_w"] = max(
                        1, int(selected_box["width"] * selected_box["scaleX"] / scale)
                    )
                    st.session_state["roi_h"] = max(
                        1, int(selected_box["height"] * selected_box["scaleY"] / scale)
                    )
                    st.session_state["roi_x"] = min(st.session_state["roi_x"], w - 1)
                    st.session_state["roi_y"] = min(st.session_state["roi_y"], h - 1)
                    st.session_state["roi_w"] = min(
                        st.session_state["roi_w"], w - st.session_state["roi_x"]
                    )
                    st.session_state["roi_h"] = min(
                        st.session_state["roi_h"], h - st.session_state["roi_y"]
                    )

        with input_col:
            st.caption("ROI coordinates")
            x = st.number_input(
                "x",
                min_value=0,
                max_value=w - 1,
                key="roi_x",
            )
            y = st.number_input(
                "y",
                min_value=0,
                max_value=h - 1,
                key="roi_y",
            )
            roi_w = st.number_input(
                "width",
                min_value=1,
                max_value=w - int(x),
                key="roi_w",
            )
            roi_h = st.number_input(
                "height",
                min_value=1,
                max_value=h - int(y),
                key="roi_h",
            )

        roi = (int(x), int(y), int(roi_w), int(roi_h))

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
