"""
Evaluate a trained YOLOv8 currency-detection model on the validation set
and print standard object-detection metrics (mAP50, mAP50-95, precision,
recall).

Usage:
    python src/evaluate.py --weights weights/best.pt --data config.yaml
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 currency detector")
    parser.add_argument("--weights", type=str, default="weights/best.pt",
                         help="Path to trained model weights")
    parser.add_argument("--data", type=str, default="config.yaml")
    return parser.parse_args()


def evaluate(args):
    model = YOLO(args.weights)
    metrics = model.val(data=args.data)

    print("\n=== Validation Metrics ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision:{metrics.box.mp:.4f}")
    print(f"Recall:   {metrics.box.mr:.4f}")


if __name__ == "__main__":
    evaluate(parse_args())
