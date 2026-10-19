# ONNX road-damage model manifest (RoadGuard AI, task 14.1)

The Model_Path (`vision/damage_detector.py`, task 14.2) loads the pinned model
named here via the OpenCV DNN module. This manifest records the **verified
source, version, and license** so the model is reproducible and attributable,
and so CI can check the on-disk artifact matches the pin. Weights are **not
committed** to the repo; they are fetched on demand (see Fetch below).

## Pinned model

| Field | Value |
| --- | --- |
| Name | road-damage-detection-yolov11 (YOLOv11s, fine-tuned) |
| Source | HuggingFace: `cvtechniques/road-damage-detection-yolov11` |
| Source URL | https://huggingface.co/cvtechniques/road-damage-detection-yolov11 |
| Base | `yolo11s.pt` (COCO-pretrained), fine-tuned on an RDD2022 subset (Roboflow) |
| Classes (4) | alligator_crack, longitudinal_crack, transverse_crack, pothole |
| License | **CC-BY-4.0** (verified from the model card metadata, 2026-xx) |
| Dataset | RDD2022 — Arya et al. (2022), multi-national road-damage dataset |
| Local artifact | `vision/models/road_damage_yolov11s.onnx` (git-ignored; fetched) |
| Expected SHA-256 | `<FILL-ON-FETCH>` — recorded by the fetch step; CI compares against it |

### Attribution (required by CC-BY-4.0)
Model: "road-damage-detection-yolov11" by cvtechniques, licensed CC-BY-4.0,
https://huggingface.co/cvtechniques/road-damage-detection-yolov11 . Trained on
RDD2022 (Arya et al., 2022). Content was rephrased for compliance with licensing
restrictions. This attribution is reproduced in the app's About/Report footer
when the Model_Path is used.

## Fetch (not committed; reproducible)

The model is distributed as PyTorch (`.pt`). OpenCV's DNN module loads **ONNX**,
so the `.pt` must be exported to ONNX once (opset 12) and placed at the local
artifact path. Documented, reproducible steps:

```bash
# 1. download the pinned weights (requires `pip install huggingface_hub`)
python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('cvtechniques/road-damage-detection-yolov11', 'best.pt'))"
# 2. export to ONNX opset 12 (requires `pip install ultralytics`)
yolo export model=best.pt format=onnx opset=12 imgsz=640
# 3. place as vision/models/road_damage_yolov11s.onnx and record its sha256
```

## OpenCV-DNN compatibility (honest caveat -> drives the 14.2 fallback)

YOLOv11 ONNX graphs are NOT guaranteed to load with `cv2.dnn.readNetFromONNX`
on every OpenCV build (newer YOLO export ops / embedded NMS can be unsupported).
Therefore the Model_Path (14.2) MUST:
  * attempt the load and raise `ModelLoadError` on failure, and
  * let the agent fall back to the Classical_Path (R4.5),
rather than claiming model detection that did not run. The manifest + fetch make
the model reproducible; the fallback keeps the system honest when a given OpenCV
build cannot load it. The Mock_Runtime backbone does not require the model.

## CI check

`vision/models/verify_model.py` verifies: (a) the manifest is well-formed and
names a license, and (b) IF the local artifact exists, its SHA-256 matches the
recorded `Expected SHA-256`. It does NOT download anything in CI and passes when
the artifact is absent (model is optional; classical path is the default).
