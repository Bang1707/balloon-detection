import cv2
import numpy as np

def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    avg_hue = int(np.mean(h))
    avg_sat = hsv[:, :, 1].mean()

    if avg_sat < 60:
        return "Not-Balloon"

    if avg_hue < 10 or avg_hue > 160:
        return "Red"
    elif 10 <= avg_hue < 25:
        return "Orange"
    elif 25 <= avg_hue < 35:
        return "Yellow"
    elif 35 <= avg_hue < 85:
        return "Green"
    elif 85 <= avg_hue < 130:
        return "Blue"
    else:
        return "Purple"