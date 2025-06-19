import os
import torch
from torchvision import transforms
from PIL import Image
import importlib


# Выбрать конфигурацию модели, вписав название
# Если вы хотите протестировать не финальную модель конфигурации, самостоятельно пропишите путь к модели (model_path) в блоке ниже
config_name = 'config_1'

config_module = importlib.import_module(f"src.configs.{config_name}")
CONFIG = config_module.CONFIG
cfg = CONFIG
MODEL_NAME = cfg['model_name']
MODEL_EXPERIMENT_NUMBER = cfg['model_experiment_number']
NUM_CLASSES = cfg['num_classes']


# Путь к весам модели
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "..", "..", "models", f"{MODEL_NAME}_{MODEL_EXPERIMENT_NUMBER}_best_model.pth")

# === Загрузка модели ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
model = model_module.get_model(NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Трансформации ===
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# === Предсказание ===
def predict(image_path, class_names=None):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_idx].item()

    if class_names:
        predicted_label = class_names[predicted_idx]
    else:
        predicted_label = str(predicted_idx)

    return predicted_label, confidence

# === Пример использования ===
if __name__ == "__main__":
    img_path = os.path.join(base_dir, "..", "..", "data", "processed", "resized_Berries_Fruit-262", "test", "barberry", "11.jpg")
    label, conf = predict(img_path)
    print(f"Предсказано: {label} с уверенностью {conf:.2f}")
