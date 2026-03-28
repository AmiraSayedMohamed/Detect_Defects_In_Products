import cv2
from ultralytics import YOLO
import os
import onnxruntime as ort
import sys

# Load model
model_path = os.path.abspath("quantized_model_uint8.onnx")
print(f"Loading model: {model_path}")

try:
    # Explicitly check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file {model_path} does not exist.")
        sys.exit(1)

    # Load ONNX model to verify compatibility
    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print("ONNX model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading ONNX model: {e}")
    sys.exit(1)

try:
    model = YOLO(model_path, task='detect')
    print("YOLO model initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing YOLO model: {e}")
    sys.exit(1)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam. Ensure it is connected and not used by another application.")
    sys.exit(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame from webcam.")
            break

        # Run YOLO detection
        try:
            results = model.predict(frame, conf=0.50, verbose=True, device='cpu')
            print("Inference completed successfully.")
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            break

        # Draw results on the frame
        annotated_frame = results[0].plot()

        # Show frame
        cv2.imshow("YOLO Detection", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"❌ Error during webcam processing: {e}")

finally:
    # Ensure resources are released
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam and windows released.")