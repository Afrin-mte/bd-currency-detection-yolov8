from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt')
    model.train(data='config.yaml', epochs=80, imgsz=640, amp=False, device='cpu')
    print("Training completed")

if __name__ == '__main__':
    train_model()