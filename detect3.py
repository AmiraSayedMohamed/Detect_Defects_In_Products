import cv2
from ultralytics import YOLO
import os
import onnxruntime as ort
import sys
import logging
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model
model_path = os.path.abspath("quantized_model_uint8.onnx")
logging.info(f"Loading model: {model_path}")

try:
    if not os.path.exists(model_path):
        logging.error(f"Model file {model_path} does not exist.")
        sys.exit(1)
    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    logging.info("ONNX model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading ONNX model: {e}")
    sys.exit(1)

try:
    model = YOLO(model_path, task='detect')
    logging.info("YOLO model initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing YOLO model: {e}")
    sys.exit(1)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Could not open webcam. Ensure it is connected and not used by another application.")
    sys.exit(1)

try:
    while True:
        # Capture frame and measure time
        start_time = time.time()
        ret, frame = cap.read()
        capture_time = (time.time() - start_time) * 1000
        logging.info(f"Frame capture time: {capture_time:.1f}ms")
        if not ret:
            logging.error("Failed to grab frame from webcam.")
            break

        # Run YOLO detection
        try:
            results = model.predict(frame, conf=0.1, iou=0.5, verbose=True, device='cpu')
            logging.info("Inference completed successfully.")
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            break

        # Draw results on the frame
        annotated_frame = results[0].plot()

        # Show frame
        cv2.imshow("YOLO Detection", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("User pressed 'q' to exit.")
            break

except Exception as e:
    logging.error(f"Webcam processing error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Webcam and windows released.")