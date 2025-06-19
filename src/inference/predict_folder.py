import os
from PIL import Image
from torchvision import transforms
import torch
import importlib
from torchvision.datasets import ImageFolder


def predict_folder(model, folder_path, transform, device, class_to_idx):
    model.eval()
    predictions = []
    correct = 0
    total = 0

    with torch.no_grad():
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if not file_name.lower().endswith((".jpg", ".png")):
                    continue

                img_path = os.path.join(root, file_name)
                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                output = model(input_tensor)
                pred_class = output.argmax(1).item()

                # Извлекаем имя папки (настоящий класс)
                class_name = os.path.basename(os.path.dirname(img_path))
                true_class = class_to_idx[class_name]

                predictions.append((file_name, pred_class, true_class))

                if pred_class == true_class:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    return predictions, accuracy




if __name__ == "__main__":

    # Выбрать конфигурацию модели, вписав название
    # Если вы хотите протестировать не финальную модель конфигурации, самостоятельно пропишите путь к модели (weights_path) ниже
    config_name = 'config_1'

    config_module = importlib.import_module(f"src.configs.{config_name}")
    CONFIG = config_module.CONFIG
    cfg = CONFIG
    MODEL_NAME = cfg['model_name']
    MODEL_EXPERIMENT_NUMBER = cfg['model_experiment_number']
    NUM_CLASSES = cfg['num_classes']

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
    model = model_module.get_model(NUM_CLASSES)

    # Путь к весам модели
    base_dir = os.path.dirname(__file__)
    weights_path = os.path.join(base_dir, "..", "..", "models", f"{MODEL_NAME}_{MODEL_EXPERIMENT_NUMBER}_best_model.pth")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(base_dir, "..", "..", "data", "processed", "resized_Berries_Fruit-262", "test")



    # Получаем соответствие "класс → индекс"
    dataset = ImageFolder(test_dir)
    class_to_idx = dataset.class_to_idx  # {'barberry': 0, 'blueberry': 1, ...}

    results, acc = predict_folder(model, test_dir, transform, DEVICE, class_to_idx)


    for fname, pred, true in results:
        print(f"{fname} -> predicted: {pred}, true: {true}")

    print(f"\nAccuracy on test set: {acc:.4f}")

