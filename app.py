# app.py
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import gradio as gr
import threading
import uvicorn

# ---------- Load ONNX model ----------
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]
IN_H, IN_W = int(input_shape[2]), int(input_shape[3])

# ---------- Helpers ----------
def preprocess(img_bgr):
    img_resized = cv2.resize(img_bgr, (IN_W, IN_H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1)) / 255.0
    return np.expand_dims(img_transposed.astype(np.float32), axis=0)

def xywh2xyxy(boxes):
    xyxy = boxes.copy()
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return xyxy

def nms(boxes, scores, iou_threshold=0.45):
    if boxes.size == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
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
    for (x1, y1, x2, y2), cls, sc in zip(boxes_xyxy, class_ids, scores):
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img_bgr, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        label = f"{int(cls)} {sc:.2f}"
        cv2.putText(img_bgr, label, (x1i, max(0, y1i - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return img_bgr

# ---------- Postprocess (universal) ----------
def postprocess(preds, orig_img_bgr, conf_thres=0.10, iou_thres=0.45):
    orig_h, orig_w = orig_img_bgr.shape[:2]
    output = preds[0]

    # Post-NMS (N,6) branch
    if isinstance(output, np.ndarray) and output.ndim == 2 and output.shape[1] == 6:
        dets = output
        if dets.size == 0:
            return orig_img_bgr, []
        boxes = dets[:, :4].astype(float)
        scores = dets[:, 4].astype(float)
        class_ids = dets[:, 5].astype(int)
        if np.max(boxes) <= 1.0:
            boxes[:, 0] *= orig_w; boxes[:, 2] *= orig_w
            boxes[:, 1] *= orig_h; boxes[:, 3] *= orig_h
        boxes_xyxy = boxes  # assume xyxy

    # Raw YOLO [1, C, num_preds] branch
    else:
        arr = np.squeeze(preds[0])    # (C, num_preds)
        arr = arr.T                   # (num_preds, C)
        if arr.size == 0:
            return orig_img_bgr, []
        boxes_xywh = arr[:, :4]
        obj_conf = arr[:, 4]
        class_scores = arr[:, 5:]
        scores_all = (obj_conf[:, None] * class_scores)
        class_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(scores_all.shape[0]), class_ids]
        mask = scores > conf_thres
        if not mask.any():
            return orig_img_bgr, []
        boxes_xywh = boxes_xywh[mask]; scores = scores[mask]; class_ids = class_ids[mask]
        boxes_xywh[:, 0] *= orig_w; boxes_xywh[:, 1] *= orig_h
        boxes_xywh[:, 2] *= orig_w; boxes_xywh[:, 3] *= orig_h
        boxes_xyxy = xywh2xyxy(boxes_xywh)

    # NMS
    keep_idx = nms(boxes_xyxy, scores, iou_threshold=iou_thres)
    if keep_idx.size == 0:
        return orig_img_bgr, []
    boxes_xyxy = boxes_xyxy[keep_idx]; scores = scores[keep_idx]; class_ids = class_ids[keep_idx]

    # Clip
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w - 1)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h - 1)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w - 1)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h - 1)

    annotated = orig_img_bgr.copy()
    annotated = draw_boxes(annotated, boxes_xyxy, class_ids, scores)

    detections = []
    for (x1, y1, x2, y2), cls, sc in zip(boxes_xyxy, class_ids, scores):
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(sc),
            "class_id": int(cls)
        })
    return annotated, detections

# ---------- FastAPI endpoint ----------
app = FastAPI()

@app.post("/predict")
async def predict_api(file: UploadFile = File(...), conf_thres: float = 0.10, iou_thres: float = 0.45):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "invalid image"})
    inp = preprocess(img_bgr)
    preds = session.run(None, {input_name: inp})
    annotated, detections = postprocess(preds, img_bgr, conf_thres=conf_thres, iou_thres=iou_thres)
    _, img_encoded = cv2.imencode('.jpg', annotated)
    return JSONResponse(content={"detections": detections, "annotated_image_bytes": img_encoded.tobytes()})

# ---------- Gradio UI with sliders ----------
def predict_gui(image, conf_thres, iou_thres):
    if image is None:
        return None
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    inp = preprocess(img_bgr)
    preds = session.run(None, {input_name: inp})
    annotated, _ = postprocess(preds, img_bgr, conf_thres=float(conf_thres), iou_thres=float(iou_thres))
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

demo = gr.Blocks()
with demo:
    gr.Markdown("## YOLO ONNX Inference — adjust thresholds")
    with gr.Row():
        img_in = gr.Image(type="numpy", label="Upload image")
        with gr.Column():
            conf_slider = gr.Slider(0.0, 1.0, value=0.10, step=0.01, label="Confidence Threshold")
            iou_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="NMS IoU Threshold")
            run_btn = gr.Button("Run Inference")
    out_img = gr.Image(type="numpy", label="Annotated image")
    run_btn.click(fn=predict_gui, inputs=[img_in, conf_slider, iou_slider], outputs=out_img)

# ---------- Run both API + UI ----------
if __name__ == "__main__":
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    threading.Thread(target=run_api, daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
