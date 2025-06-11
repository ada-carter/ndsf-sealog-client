import argparse
import csv
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None


def run_inference(image_paths, weights, conf=0.25, iou=0.7):
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")
    model = YOLO(weights)
    predictions = []
    for img_path in image_paths:
        results = model(img_path, conf=conf, iou=iou)
        if not results:
            continue
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box[:6]
            predictions.append({
                'image': str(img_path),
                'class': int(cls),
                'score': float(score),
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2)
            })
    return predictions


def write_csv(predictions, output_file):
    if not predictions:
        return
    fieldnames = predictions[0].keys()
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv11 inference on images")
    parser.add_argument('--weights', required=True, help='Path to .pt weights file')
    parser.add_argument('--images', nargs='+', required=True, help='Image files or directories')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold')
    parser.add_argument('--output', default='predictions.csv', help='CSV output file')
    args = parser.parse_args()

    img_files = []
    for path in args.images:
        p = Path(path)
        if p.is_dir():
            img_files.extend([f for f in p.glob('**/*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        else:
            img_files.append(p)

    preds = run_inference(img_files, args.weights, conf=args.conf, iou=args.iou)
    write_csv(preds, args.output)


if __name__ == '__main__':
    main()
