import tf2onnx
import onnx

onnx_model = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, "model.onnx")
