import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Path to your original ONNX model
model_input = "best.onnx"

# Path where the new quantized model will be saved
model_output = "quantized_model_uint8.onnx"

# Use QUInt8 instead of QInt8
quantize_dynamic(model_input, model_output, weight_type=QuantType.QUInt8)

print(f"Quantized model saved to: {model_output}")