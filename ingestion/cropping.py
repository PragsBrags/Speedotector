import cv2

def cropping_selection(cap):
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video")
        cap.release()
        exit()

    rects = cv2.selectROIs("Select Static Zones", first_frame, fromCenter=False)
    return rects

def ROI_cropping_all_frames(frame, rects):
    # Loop over the selected bounding boxes and crop them
    crops = []
    h_frame, w_frame = frame.shape[:2]
    for rect in rects:
        x, y, w, h = map(int, rect)
        # clamp to frame bounds
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        crops.append(frame[y:y+h, x:x+w])
    return crops