import cv2
import time


object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16)


def calculate_sharpness(img):
    #Calculates the Laplacian variance to measure blurriness
    return cv2.Laplacian(img, cv2.CV_64F).var()

def frame(cap) :
    # Threshold for motion detection
    THRESHOLD = 100
    best_frame = None
    max_score = -1
    frames_since_motion = 0
    COOLDOWN_LIMIT = 10

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            return
        
        mask = object_detector.apply(frame)

        contours, _ =cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        active_motion = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > THRESHOLD:
                active_motion = True
                score = calculate_sharpness(frame)
                final_score = score * area

                if final_score > max_score:
                    max_score = final_score
                    best_frame = frame.copy()
                
                frames_since_motion = 0
        
        if active_motion and best_frame is not None:
            frames_since_motion = frames_since_motion + 1

            if frames_since_motion > COOLDOWN_LIMIT:
                yield best_frame
                best_frame = None
                max_score = -1


