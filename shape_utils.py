import cv2
import numpy as np
import math

def detect_shape(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Unknown"

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)

    if peri == 0:
        return "Unknown"

    if area < 1500:
        return "Ignored"    

    circularity = 4 * math.pi * area / (peri * peri)

    if circularity > 0.75:
        return "Round"
    elif circularity > 0.45:
        return "Oval"
    else:
        return "Irregular"
