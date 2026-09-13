import cv2
from ultralytics import YOLO
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Path to your trained model weights
model_path = r"C:\Users\LENOVO\PycharmProjects\PythonProject4\runs\detect\train54\weights\best.pt"

# Load the trained YOLOv8 model
model = YOLO(model_path)

# Open the webcam (default camera index is 0)
cap = cv2.VideoCapture(0)  # Use 1 or other index if you have multiple cameras

# Set webcam resolution (optional)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

print("Starting webcam. Press 'q' to exit.")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Perform inference with YOLOv8
    results = model(frame)

    # Extract detected objects
    detected_objects = results[0].boxes.cls  # Extract class indices
    class_names = results[0].names           # Extract class names

    # Generate audio feedback for detections
    if detected_objects is not None:
        detected_classes = [class_names[int(cls_idx)] for cls_idx in detected_objects]
        for detected_class in detected_classes:
            feedback = f"Detected {detected_class}"
            print(feedback)  # Print the feedback to the console
            engine.say(feedback)  # Convert the feedback to speech
            engine.runAndWait()

    # Visualize the detection results
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("Currency Detection", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam resources and close the display window
cap.release()
cv2.destroyAllWindows()

