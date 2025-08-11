import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import gradio as gr
import io
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# Get input name & shape
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

def preprocess(img):
    img_resized = cv2.resize(img, (input_shape[2], input_shape[3]))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1)) / 255.0
    img_tensor = np.expand_dims(img_transposed.astype(np.float32), axis=0)
    return img_tensor

def postprocess(preds, img, conf_thres=0.3):
    """
    Handles:
    1. Post-NMS: (N, 6) => x1, y1, x2, y2, conf, class_id
    2. Raw YOLO: (1, 4 + 1 + num_classes, num_preds)
    """
    output = preds[0]
    annotated = img.copy()

    # Case 1: Post-NMS format
    if output.ndim == 2 and output.shape[1] == 6:
        for x1, y1, x2, y2, score, cls_id in output:
            if score > conf_thres:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"cls:{int(cls_id)} {score:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

    # Case 2: Raw YOLO ONNX format [1, 4+1+num_classes, num_preds]
    elif output.ndim == 3:
        output = np.squeeze(output).T  # shape: (num_preds, 4+1+num_classes)
        boxes = output[:, :4]
        obj_conf = output[:, 4]
        class_scores = output[:, 5:]
        scores = obj_conf[:, None] * class_scores
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        for (x, y, w, h), score, cls_id in zip(boxes, confidences, class_ids):
            if score > conf_thres:
                x1, y1, x2, y2 = map(int, [x - w/2, y - h/2, x + w/2, y + h/2])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"cls:{cls_id} {score:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

    return annotated
    
# API
app = FastAPI()

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    inp = preprocess(img)
    preds = session.run(None, {input_name: inp})
    return JSONResponse(content={"predictions": preds[0].tolist()})

# GUI
def predict_gui(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    inp = preprocess(img_bgr)
    preds = session.run(None, {input_name: inp})
    annotated = postprocess(preds, img_bgr)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

demo = gr.Interface(
    fn=predict_gui,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="YOLO ONNX Inference"
)

if __name__ == "__main__":
    import threading
    import uvicorn

    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    threading.Thread(target=run_api).start()
    demo.launch(server_name="0.0.0.0", server_port=7860)
