import cv2
from ultralytics import YOLO
import pytesseract

# Path to your trained model
model = YOLO("best.onnx")

# Expected printed text (replace with your ground truth)
expected_text = "dasani"

# Open webcam or video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model.predict(frame, conf=0.50, verbose=False, device='cpu')

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])  # class id
            label = model.names[cls_id]  # class name

            # If the detection is a label
            if label == "label":
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]

                # Run OCR
                text = pytesseract.image_to_string(roi).strip()

                # Compare
                if expected_text.lower() in text.lower():
                    status = "✅ Good Label"
                    color = (0, 255, 0)  # Green
                else:
                    status = "❌ Damaged Label"
                    color = (0, 0, 255)  # Red

                # Draw bounding box + OCR text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{status}: {text}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
