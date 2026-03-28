from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("best.pt")

# Export to ONNX (good for Raspberry Pi)
model.export(format="onnx", opset=11, dynamic=True, simplify=True)
