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
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        
        # Crop the bounding box area
        return frame[y:y+h, x:x+w]