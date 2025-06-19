import os
import torch
from tqdm import tqdm
import importlib
import torch.nn as nn
import torch.optim as optim


# Выбрать конфигурацию модели, вписав название
config_name = 'config_1'

config_module = importlib.import_module(f"src.configs.{config_name}")
MODEL_SAVE_PATH = config_module.MODEL_SAVE_PATH
CONFIG = config_module.CONFIG


def train():
    cfg = CONFIG
    model_module = importlib.import_module(f"src.models.{cfg['model_name']}")
    model = model_module.get_model(cfg["num_classes"]).to(cfg["device"])

    dataloader_module = importlib.import_module(f"src.data_loaders.{cfg['data_loader_name']}")
    get_dataloaders = dataloader_module.get_dataloaders

    train_loader, val_loader = get_dataloaders(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "resized_Berries_Fruit-262"), cfg["batch_size"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(MODEL_SAVE_PATH, f"{cfg['model_name']}_{cfg['model_experiment_number']}_best_model.pth")

    for epoch in range(cfg["num_epochs"]):
        print(f"\n📘 Epoch {epoch+1}/{cfg['num_epochs']}")
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(cfg["device"]), labels.to(cfg["device"])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total

        # === Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(cfg["device"]), labels.to(cfg["device"])
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / total
        val_acc = correct / total

        print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print("Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                print("Early stopping.")
                break

    print("Обучение завершено.")
