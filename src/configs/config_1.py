import os
import torch

# === Пути ===
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed", "resized_Berries_Fruit-262")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "..", "models")

# === Настройки обучения ===
CONFIG = {
    "data_loader_name": "dataLoader1",
    "model_name": "efficientnetB1",
    "model_experiment_number": 1.0,
    "num_classes": 44,
    "batch_size": 32,
    "num_epochs": 30,
    "patience": 5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "lr": 1e-4,
    "weight_decay": 1e-4
}
