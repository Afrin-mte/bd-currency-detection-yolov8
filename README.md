#  Bangladeshi Currency Detection using YOLOv8

Real-time detection and classification of Bangladeshi Taka banknotes from a
live webcam feed, with spoken audio feedback for accessibility. Built with a
custom-trained **YOLOv8** object detector.

![Python](https://img.shields.io/badge/Python-3.9%2B)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Realtime)
![License](https://img.shields.io/badge/License-MIT)

<!--
  📸 ADD YOUR DEMO HERE — this is the single highest-impact thing you can do
  for this README. Record a 10–15s screen capture of the webcam window
  detecting a note, convert it to a GIF (e.g. with ezgif.com or ScreenToGif),
  save it to docs/assets/demo.gif, and this will render automatically.
-->
![Demo](docs/assets/demo.gif)

## 📖 Overview

This project trains a YOLOv8 object-detection model to recognize Bangladeshi
Taka notes (৳1 to ৳1000) from images or a live camera feed, then reads the
detected denomination aloud using text-to-speech. It was built as a group
project with accessibility in mind — helping visually-impaired users identify
currency notes independently.

**Key features**

- Custom-trained YOLOv8 model on a hand-labeled dataset of Taka note images
- Real-time inference from a webcam with bounding-box + confidence overlay
- Text-to-speech announcements of detected denominations, with cooldown
  logic to avoid repeating the same announcement every frame
- A secondary generic-object detection demo (COCO classes) for pipeline
  testing without the custom weights
- Standalone evaluation script reporting mAP50, mAP50-95, precision, and
  recall on the validation set

## 🧠 Tech Stack

| Component        | Tool |
|-------------------|------|
| Object detection  | [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) |
| Computer vision   | OpenCV |
| Text-to-speech    | pyttsx3 |
| Language          | Python 3.9+ |

## 🗂️ Project Structure

```
bd-currency-detection-yolov8/
├── src/
│   ├── train.py                   # Train YOLOv8 on the currency dataset
│   ├── evaluate.py                # Compute mAP / precision / recall
│   ├── detect_webcam_voice.py     # Real-time detection + TTS feedback
│   └── detect_webcam_generic.py   # Generic COCO-class webcam demo
├── config.yaml                    # Dataset + class definitions
├── docs/assets/                   # Demo GIF, sample detections, plots
├── requirements.txt
├── LICENSE
└── README.md
```

## 📊 Dataset

The model is trained on a custom-labeled dataset of Bangladeshi Taka note
images covering 11 classes (1, 2, 5, 10, 20, 50, 100, 500, and 1000 Taka,
plus an "unclassified currency" catch-all). Images were annotated in YOLO
format (one `.txt` per image with `class x_center y_center width height`).

> The raw dataset isn't included in this repo due to size. If you're
> reproducing this project, organize your images as:
> ```
> data/currency/train/images, data/currency/train/labels
> data/currency/test/images,  data/currency/test/labels
> ```
> and update the paths in `config.yaml` accordingly.

<!-- Optional: add 2–3 sample annotated training images here, e.g.
![Sample annotations](docs/assets/sample_annotations.jpg)
-->

## 📈 Results

| Metric      | Score |
|-------------|-------|
| mAP@0.5     | _fill in from your `evaluate.py` run_ |
| mAP@0.5:0.95| _fill in_ |
| Precision   | _fill in_ |
| Recall      | _fill in_ |

<!--
  Run `python src/evaluate.py --weights weights/best.pt` and paste the
  numbers above. Ultralytics also auto-saves a confusion matrix and
  results.png under runs/detect/val*/ after training — copy those into
  docs/assets/ and embed them here, e.g.:
  ![Confusion Matrix](docs/assets/confusion_matrix.png)
  ![Training Curves](docs/assets/results.png)
-->

## 🚀 Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/bd-currency-detection-yolov8.git
cd bd-currency-detection-yolov8
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py --data config.yaml --epochs 80 --imgsz 640 --device cpu
```

Trained weights are saved to `runs/detect/train/weights/best.pt`. Copy the
final weights to `weights/best.pt` for use with the other scripts.

### 3. Evaluate

```bash
python src/evaluate.py --weights weights/best.pt --data config.yaml
```

### 4. Run real-time detection with voice feedback

```bash
python src/detect_webcam_voice.py --weights weights/best.pt --camera 0
```

Press `q` to quit the webcam window.

## 🔮 Future Improvements

- Expand the dataset with more lighting conditions, note wear, and angles
- Deploy as a lightweight mobile app (e.g. TFLite / ONNX export) for
  offline, on-device use
- Multi-currency support
- Fine-grained detection of counterfeit indicators

## 🙌 Acknowledgments

Built as a group university project exploring applied computer vision for
accessibility. Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).

## 📄 License

Released under the [MIT License](LICENSE).
