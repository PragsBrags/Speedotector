import cv2

from ingestion.cropping import cropping_selection


def calculate_sharpness(img):
    # Calculates the Laplacian variance to measure blurriness
    return cv2.Laplacian(img, cv2.CV_64F).var()


def _clip_rect(rect, frame_shape):
    h_frame, w_frame = frame_shape[:2]
    x, y, w, h = map(int, rect)
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))
    return x, y, w, h


def _crop_with_rect(frame, rect):
    x, y, w, h = _clip_rect(rect, frame.shape)
    return frame[y : y + h, x : x + w], (x, y, w, h)


def selected_frames(cap, roi=None):
    # Threshold for motion detection
    THRESHOLD = 100
    best_frame = None
    best_frame_number = None
    best_roi = None
    max_score = -1
    frames_since_motion = 0
    COOLDOWN_LIMIT = 10
    frame_number = -1
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16)

    rects = cropping_selection(cap) if roi is None else [roi]

    if rects is None:
        return

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            if best_frame is not None:
                yield {
                    "image": best_frame,
                    "frame_number": best_frame_number,
                    "roi": best_roi,
                }
            return

        frame_number += 1
        active_motion = False

        for rect in rects:
            cropped_frame, clipped_roi = _crop_with_rect(current_frame, rect)
            mask = object_detector.apply(cropped_frame)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > THRESHOLD:
                    active_motion = True
                    score = calculate_sharpness(cropped_frame)
                    final_score = score * area

                    if final_score > max_score:
                        max_score = final_score
                        best_frame = cropped_frame.copy()
                        best_frame_number = frame_number
                        best_roi = clipped_roi

        if active_motion:
            frames_since_motion = 0
            continue

        frames_since_motion += 1
        if best_frame is not None and frames_since_motion > COOLDOWN_LIMIT:
            yield {
                "image": best_frame,
                "frame_number": best_frame_number,
                "roi": best_roi,
            }
            best_frame = None
            best_frame_number = None
            best_roi = None
            max_score = -1


def frame(cap, roi=None):
    for selected_frame in selected_frames(cap, roi=roi):
        yield selected_frame["image"]
