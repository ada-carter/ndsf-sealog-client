# NDSF Sealog Client

This repository contains the web frontend for the Sealog event logging software used by the [National Deep Submergence Facility][ndsf] with the vehicles ROV *Jason* and HOV *Alvin*.

The upstream repository can be found at [OceanDataTools/sealog-client-vehicle][upstream].

## YOLOv11 inference utility

This repository includes a small helper script under `tools/` for running
YOLOv11 (via the [ultralytics](https://github.com/ultralytics/ultralytics)
package) against one or more images. The script accepts a path to a local `.pt`
weights file and outputs detections to a CSV file.

```bash
python3 tools/yolov11_inference.py --weights model.pt --images image_folder \
    --conf 0.25 --iou 0.7 --output predictions.csv
```

### Running inside the UI

Start the lightweight server that exposes the inference helper:

```bash
npm run yolo-server
```

An "Run YOLOv11 Detection" option will appear on the **Tasks** page. It allows
entering the weights and image paths, then downloads the CSV results.


  [ndsf]: https://ndsf.whoi.edu/
  [upstream]: https://github.com/oceandatatools/sealog-client-vehicle/
