# 💵 Bangladeshi Currency Detection using YOLOv8

Real-time detection and classification of Bangladeshi Taka banknotes from a
live webcam feed, with spoken audio feedback for accessibility. Built with a
custom-trained **YOLOv8** object detector.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Realtime-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

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
plus a duplicate "500 Taka"-style class and an "unclassified currency"
catch-all — see [Limitations](#-limitations--honest-notes)). Images were
annotated in YOLO format (one `.txt` per image with
`class x_center y_center width height`).

Here's a batch of training images with their ground-truth boxes:

![Training batch sample](docs/assets/train_batch_sample.jpg)

> The raw dataset isn't included in this repo due to size. If you're
> reproducing this project, organize your images as:
> ```
> data/currency/train/images, data/currency/train/labels
> data/currency/test/images,  data/currency/test/labels
> ```
> and update the paths in `config.yaml` accordingly.

## 📈 Results

Trained for 20 epochs on CPU. Final validation metrics (see [`docs/results.csv`](docs/results.csv) for the full per-epoch log):

| Metric       | Score  |
|--------------|--------|
| Precision    | 0.706  |
| Recall       | 0.596  |
| mAP@0.5      | 0.648  |
| mAP@0.5:0.95 | 0.332  |

**Training curves** (loss and metrics over all 20 epochs):

![Training curves](docs/assets/training_curves.png)

**Precision / Recall / PR curves per class:**

| Precision–Confidence | Recall–Confidence |
|---|---|
| ![Precision curve](docs/assets/precision_curve.png) | ![Recall curve](docs/assets/recall_curve.png) |

![PR curve](docs/assets/pr_curve.png)

**Sample predictions vs. ground truth** on a held-out validation batch
(left: ground truth, right: model predictions with confidence scores):

| Ground Truth | Predictions |
|---|---|
| ![Val labels](docs/assets/val_batch0_labels.jpg) | ![Val predictions](docs/assets/val_batch0_pred.jpg) |

## ⚠️ Limitations & Honest Notes

This is a learning project, and the numbers above tell an honest story —
worth stating plainly rather than glossing over:

- **Per-class performance is very uneven.** Common, high-contrast notes
  (Twenty: 0.90 AP, currency-generic: 0.91 AP, Fifty: 0.85 AP) detect well.
  Two classes perform poorly: **"500 taka" (AP 0.163)** and **"Five Hundred
  taka" (AP 0.000)** — these two labels almost certainly refer to the same
  physical note under slightly different names, which splits and confuses
  the training signal. **"two taka" also underperforms** noticeably.
- **Likely causes:** overlapping/duplicate class definitions (see
  `config.yaml`), fewer training images for the weak classes, and visually
  similar notes (worn, faded, or photographed at odd angles — see the
  training batch sample above) that are hard to tell apart even by eye.
- **Fix before relying on this for anything real:** merge or relabel the
  duplicate 500-taka classes, audit class balance, and add more images for
  underrepresented notes.
- Trained on CPU for only 20 epochs — a GPU + more epochs would very
  likely close much of this gap.

Calling this out isn't a weakness of the writeup — it demonstrates the
ability to read evaluation curves critically instead of only reporting the
headline mAP.

## 🚀 Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/bd-currency-detection-yolov8.git
cd bd-currency-detection-yolov8
pip install -r requirements.txt
```

### 2. (Optional) Train from scratch

A trained checkpoint is already included at `weights/best.pt`, so you can
skip straight to step 3 or 4. To retrain from scratch instead:

```bash
python src/train.py --data config.yaml --epochs 80 --imgsz 640 --device cpu
```

Trained weights are saved to `runs/detect/train/weights/best.pt`. Copy the
final weights to `weights/best.pt` to use them with the other scripts.

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
