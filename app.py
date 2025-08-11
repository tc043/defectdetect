import gradio as gr
from ultralytics import YOLO

model = YOLO("best.pt")

def predict_image(image):
    results = model.predict(image, conf=0.3)
    annotated_img = results[0].plot()  # Draw boxes on image
    return annotated_img

# Build GUI
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload an image"),
    outputs=gr.Image(type="numpy", label="Detected objects"),
    title="YOLO Inference Demo",
    description="Upload an image to run object detection."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
