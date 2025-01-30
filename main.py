import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

model = YOLO("yolo11s.pt")
names = model.model.names
cap = cv2.VideoCapture('output.mp4')

allowed_classes = {0: 'humain', 1: 'Bike', 2: 'voiture'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    results = model.track(frame, persist=True)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            if class_id in allowed_classes:
                label = f'{allowed_classes[class_id]} ({conf:.2f})'
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, label, (x1, y1 - 10), 1, 1)

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
