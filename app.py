import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from pipeline import process_video

st.set_page_config(page_title="Speedotector", layout="wide")

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

    st.video(video_path)

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        st.error("Could not read first frame from video.")
        st.stop()

    h, w = first_frame.shape[:2]

    st.subheader("ROI settings")

    if use_full_frame:
        roi = (0, 0, w, h)
        st.info(f"Using full frame: x=0, y=0, width={w}, height={h}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        x = col1.number_input("x", min_value=0, max_value=w - 1, value=0)
        y = col2.number_input("y", min_value=0, max_value=h - 1, value=0)
        roi_w = col3.number_input("width", min_value=1, max_value=w, value=w)
        roi_h = col4.number_input("height", min_value=1, max_value=h, value=h)

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
