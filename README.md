# YOLO ONNX Object Detection API & UI

This project provides a simple and easy-to-use API and web interface for performing object detection using a YOLO ONNX model. The application is built with FastAPI for the API and Gradio for the user interface, and it can be easily deployed using Docker.

## Features

-   **FastAPI Backend:** A robust and fast backend for serving the object detection model.
-   **Gradio UI:** An interactive web interface for uploading images and adjusting inference parameters.
-   **ONNX Runtime:** Utilizes the high-performance ONNX Runtime for model inference.
-   **Dockerized:** Comes with a `Dockerfile` for easy and reproducible deployment.
-   **Adjustable Thresholds:** Allows for real-time adjustment of confidence and IoU thresholds.

## How to Run

You can run this application either using Docker (recommended) or by setting up a local Python environment.

### Using Docker

**Prerequisites:**
- Docker installed on your machine.

1.  **Build the Docker image:**
    ```bash
    docker build -t yolo-onnx-app .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 7860:7860 -p 8000:8000 yolo-onnx-app
    ```

### Local Setup

**Prerequisites:**
- Python 3.10 or later.
- The dependencies listed in `requirements.txt`.

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the application:**
    ```bash
    python app.py
    ```

## Accessing the Application

Once the application is running, you can access the following:

-   **Gradio UI:** Open your web browser and navigate to `http://localhost:7860`.
-   **FastAPI Docs:** For API documentation, go to `http://localhost:8000/docs`.

## API Endpoint

### `/predict`

This endpoint accepts an image file and returns the detected objects and an annotated image.

-   **Method:** `POST`
-   **Form Data:**
    -   `file`: The image file to be processed.
    -   `conf_thres` (optional): Confidence threshold for object detection (default: `0.10`).
    -   `iou_thres` (optional): IoU threshold for non-maximum suppression (default: `0.45`).
-   **Success Response:**
    -   **Code:** `200 OK`
    -   **Content:**
        ```json
        {
          "detections": [
            {
              "bbox": [x1, y1, x2, y2],
              "score": 0.95,
              "class_id": 0
            }
          ],
          "annotated_image_bytes": "..."
        }
        ```
-   **Error Response:**
    -   **Code:** `400 Bad Request`
    -   **Content:**
        ```json
        {"error": "invalid image"}
        ```

## Dependencies

The project relies on the following Python libraries:

-   `onnxruntime`
-   `opencv-python`
-   `numpy`
-   `fastapi`
-   `uvicorn`
-   `gradio`
-   `python-multipart`
-   `pillow`

These can be found in the `requirements.txt` file.
