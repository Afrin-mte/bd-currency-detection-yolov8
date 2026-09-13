from ultralytics import YOLO
import cv2
allowed_classes = ["bottle", "laptop", "cell phone", "book", "cup"]
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw bounding boxes and filter classes
    annotated_frame = frame.copy()
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            cls_name = r.names[cls_id]

            if cls_name in allowed_classes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw rectangle
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Label with class name and confidence
                label = f"{cls_name} {conf*100:.1f}%"
                cv2.putText(annotated_frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
