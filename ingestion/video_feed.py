import cv2


object_detector = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=16
)


def calculate_sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def frame(cap):
    THRESHOLD = 100
    COOLDOWN_LIMIT = 10

    best_frame = None
    max_score = -1
    frames_since_motion = 0

    while cap.isOpened():
        ret, frame_data = cap.read()

        if not ret:
            print("End of video.")
            break

        mask = object_detector.apply(frame_data)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        active_motion = False

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > THRESHOLD:
                active_motion = True

                score = calculate_sharpness(frame_data)
                final_score = score * area

                if final_score > max_score:
                    max_score = final_score
                    best_frame = frame_data.copy()

        if active_motion:
            frames_since_motion = 0
        else:
            frames_since_motion += 1

        if best_frame is not None and frames_since_motion > COOLDOWN_LIMIT:
            yield best_frame
            best_frame = None
            max_score = -1
            frames_since_motion = 0