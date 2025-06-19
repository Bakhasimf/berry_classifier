import os
from PIL import Image
from torchvision import transforms

# Размер изображений
TARGET_SIZE = (224, 224)

# Исходная папка с данными
base_dir = os.path.dirname(__file__)
dataset_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "data", "processed", "Berries_Fruit-262"))

# Папка для сохранения ресайзнутых копий (новая)
# dataset_dir_resized = dataset_dir + "_resized"
dataset_dir_resized = os.path.join(dataset_dir, "..", "resized_Berries_Fruit-262")


splits = ["train", "val", "test"]

# Создаём трансформ для ресайза (torchvision)
resize_transform = transforms.Resize(TARGET_SIZE)


def resize_image_torchvision(image_path, save_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_resized = resize_transform(img)

            # Создаем директорию, если её нет
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            img_resized.save(save_path)
            print(f"Сохранено: {save_path}")
    except Exception as e:
        print(f"[Ошибка] {image_path}: {e}")


def resize_split(split_name):
    split_path = os.path.join(dataset_dir, split_name)
    split_resized_path = os.path.join(dataset_dir_resized, split_name)
    if not os.path.exists(split_path):
        print(f"[Пропущено] Нет директории: {split_path}")
        return

    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        class_resized_path = os.path.join(split_resized_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            save_path = os.path.join(class_resized_path, filename)
            resize_image_torchvision(file_path, save_path)



for split in splits:
    print(f"→ Обработка {split}...")
    resize_split(split)
print("✅ Все изображения приведены к 224x224 и сохранены в новую папку.")
