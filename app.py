# app.py
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import gradio as gr
import threading
import uvicorn
from PIL import Image
import io

# ---------- Load ONNX model ----------
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]
IN_H, IN_W = int(input_shape[2]), int(input_shape[3])

# ---------- Helpers ----------
def preprocess(img_bgr):
    """Resize to model input and normalize (0-1), return float32 tensor (1,3,H,W)."""
    img_resized = cv2.resize(img_bgr, (IN_W, IN_H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1)) / 255.0
    img_tensor = np.expand_dims(img_transposed.astype(np.float32), axis=0)
    return img_tensor

def xywh2xyxy(boxes):
    """boxes: (N,4) in x_center,y_center,w,h -> return x1,y1,x2,y2"""
    xyxy = boxes.copy()
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return xyxy

def nms(boxes, scores, iou_threshold=0.45):
    """
    Greedy NMS. boxes: (N,4) xyxy, scores: (N,)
    Returns indices of kept boxes.
    """
    if boxes.size == 0:
        return np.array([], dtype=int)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)

def draw_boxes(img_bgr, boxes_xyxy, class_ids, scores):
    """Draw boxes on image (in-place) and return annotated image."""
    for (x1, y1, x2, y2), cls, sc in zip(boxes_xyxy, class_ids, scores):
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img_bgr, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        label = f"{int(cls)} {sc:.2f}"
        cv2.putText(img_bgr, label, (x1i, max(0, y1i - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return img_bgr

# ---------- Postprocess (universal) ----------
def postprocess(preds, orig_img_bgr, conf_thres=0.10, iou_thres=0.45):
    """
    Supports:
     - Post-NMS: shape (N,6) -> x1,y1,x2,y2,conf,cls_id (either normalized or pixels)
     - Raw YOLO: shape [1, 4+1+num_classes, num_preds] (e.g. [1,11,8400])
    Returns: annotated image (BGR), and list of detections dicts
    """
    orig_h, orig_w = orig_img_bgr.shape[:2]
    output = preds[0]

    # Case A: Post-NMS final outputs (N,6)
    if isinstance(output, np.ndarray) and output.ndim == 2 and output.shape[1] == 6:
        dets = output
        if dets.size == 0:
            return orig_img_bgr, []
        boxes = dets[:, :4].astype(float)
        scores = dets[:, 4].astype(float)
        class_ids = dets[:, 5].astype(int)

        # Detect whether boxes are normalized (<=1) or already pixel coords (>1)
        if np.max(boxes) <= 1.0:
            boxes[:, 0] *= orig_w
            boxes[:, 2] *= orig_w
            boxes[:, 1] *= orig_h
            boxes[:, 3] *= orig_h

        boxes_xyxy = boxes  # assume already xyxy

    # Case B: Raw YOLO output [1, 4+1+num_classes, num_preds]
    else:
        arr = np.squeeze(preds[0])
        arr = arr.T  # (num_preds, 4+1+num_classes)

        if arr.size == 0:
            return orig_img_bgr, []

        boxes_xywh = arr[:, :4]  # (N,4)
        obj_conf = arr[:, 4]     # (N,)
        class_scores = arr[:, 5:] # (N, num_classes)

        # combine
        scores_all = (obj_conf[:, None] * class_scores)  # (N, num_classes)
        class_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(scores_all.shape[0]), class_ids]

        # filter by conf threshold
        mask = scores > conf_thres
        if not mask.any():
            return orig_img_bgr, []
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # convert to xyxy in original image coords
        boxes_xywh[:, 0] *= orig_w   # x center
        boxes_xywh[:, 1] *= orig_h   # y center
        boxes_xywh[:, 2] *= orig_w   # width
        boxes_xywh[:, 3] *= orig_h   # height

        boxes_xyxy = xywh2xyxy(boxes_xywh)

    # At this point boxes_xyxy, class_ids, scores available
    # Run NMS
    keep_idx = nms(boxes_xyxy, scores, iou_threshold=iou_thres)
    if keep_idx.size == 0:
        return orig_img_bgr, []

    boxes_xyxy = boxes_xyxy[keep_idx]
    scores = scores[keep_idx]
    class_ids = class_ids[keep_idx]

    # Clip boxes to image
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w - 1)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h - 1)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w - 1)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h - 1)

    # Draw boxes
    annotated = orig_img_bgr.copy()
    annotated = draw_boxes(annotated, boxes_xyxy, class_ids, scores)

    # prepare detections list for JSON
    detections = []
    for (x1, y1, x2, y2), cls, sc in zip(boxes_xyxy, class_ids, scores):
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(sc),
            "class_id": int(cls)
        })

    return annotated, detections

# ---------- API ----------
app = FastAPI()

@app.post("/predict")
async def predict_api(file: UploadFile = File(...), conf_thres: float = 0.10):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "invalid image"})

    inp = preprocess(img_bgr)
    preds = session.run(None, {input_name: inp})
    annotated, detections = postprocess(preds, img_bgr, conf_thres=conf_thres)

    # encode annotated image to JPEG for convenience (raw bytes)
    _, img_encoded = cv2.imencode('.jpg', annotated)
    return JSONResponse(content={
        "detections": detections,
        "annotated_image_bytes": img_encoded.tobytes()
    })

# ---------- GUI ----------
def predict_gui(image):
    # Gradio passes image as RGB numpy (H,W,3)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    inp = preprocess(img_bgr)
    preds = session.run(None, {input_name: inp})
    annotated, _ = postprocess(preds, img_bgr, conf_thres=0.10)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

demo = gr.Interface(
    fn=predict_gui,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="YOLO ONNX Inference"
)

if __name__ == "__main__":
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    threading.Thread(target=run_api, daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
