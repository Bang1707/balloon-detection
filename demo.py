import cv2
from ultralytics import YOLO

#utils
from color_utils import detect_color
from shape_utils import detect_shape

# trained model
model = YOLO("runs/detect/train/weights/best.pt")

# webcam (0 = default cam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame, conf=0.55, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            # bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # güvenlik: boş ROI olmasın
            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]

            # color & shape
            try:
                color = detect_color(roi)
                shape = detect_shape(roi)
            except:
                color = "Unknown"
                shape = "Unknown"


            if color == "Not-Balloon":
                continue
            if shape == "Ignored":
                continue
            
            label = f"{color} {shape} | {conf:.2f}"

            # çizimler
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Balloon Detection Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
