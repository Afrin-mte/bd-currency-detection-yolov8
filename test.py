from ultralytics import YOLO

# Load the trained model
model = YOLO("best.pt")

# Show training results if available
metrics = model.metrics  # This works if training history is saved
print(metrics)

# Or evaluate on validation set to get metrics
results = model.val()
print(results)

