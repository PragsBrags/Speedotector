import cv2

from ingestion.util import get_limits


class TrafficLightDetection:
    def __init__ (self, red_ratio_threshold = 0.03):
        self.red = [0,0,255]
        self.red_ratio_threshold = red_ratio_threshold

    
    def is_red(self, frame, light_region) :
        x,y,w,h = map(int, light_region)
        crop = frame[y:y+h, x:x+w]

        hsv_frame = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        lower_limit, upper_limit = get_limits(color = self.red)

        mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)

        red_pixels = cv2.countNonZero(mask)
        total_pixels = crop.shape[0] * crop.shape[1]
        red_ratio = red_pixels / total_pixels

        return red_ratio > self.red_ratio_threshold