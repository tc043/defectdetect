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
    # preds[0] shape: (num_detections, attributes)
    output = preds[0]
    annotated = img.copy()

    for det in output:
        x1, y1, x2, y2 = det[:4]
        object_conf = det[4]
        class_scores = det[5:]

        cls_id = int(np.argmax(class_scores))
        cls_conf = class_scores[cls_id]

        # Final confidence = objectness * class confidence
        score = object_conf * cls_conf

        if score > conf_thres:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated, f"cls:{cls_id} {score:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
