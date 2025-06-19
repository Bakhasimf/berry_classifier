import torch
import importlib

MODEL_NAME = 'efficientnetB1'
NUM_CLASSES = 44
WEIGHTS_PATH = "efficientnetB1_1.0_best_model.pth"
EXPORT_PATH = "efficientnetB1_1.0_best_model.onnx"

model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
model = model_module.get_model(NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 240, 240)
torch.onnx.export(
    model, dummy_input, EXPORT_PATH,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print("Model exported to ONNX.")
