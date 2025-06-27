from fastapi import FastAPI, UploadFile, File
from PIL import Image
import onnxruntime as ort
import numpy as np
from torchvision import transforms
import io
import logging.config
import yaml
import os
from services.database.crud import save_prediction
from fastapi.responses import JSONResponse


app = FastAPI()

# === Логгирование ===
config_path = os.path.join(os.path.dirname(__file__), "..", "..", "logging_config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
logging.config.dictConfig(config)
logger = logging.getLogger("api")

# === Модель ===
base_dir = os.path.dirname(__file__)
ONNX_MODEL_PATH = os.path.join(base_dir, "..", "..", "models", "efficientnetB1_1.0_best_model.onnx")

try:
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    logger.info(f"ONNX модель успешно загружена из {ONNX_MODEL_PATH}")
except Exception as e:
    logger.exception(f"Ошибка при загрузке ONNX модели: {e}")
    raise

# === Трансформ ===
transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = [
    ("barberry", "барбарис"),
    ("bayberry", "восковница"),
    ("bearberry", "толокнянка"),
    ("black berry", "ежевика"),
    ("black cherry", "черная вишня"),
    ("blueberry", "голубика"),
    ("buffaloberry", "буффало-ягода"),
    ("burmese grape", "бирманский виноград"),
    ("cape gooseberry", "перуанская физалис"),
    ("cedar bay cherry", "кедровая вишня"),
    ("ceylon gooseberry", "цейлонский крыжовник"),
    ("cherry", "вишня"),
    ("chokeberry", "арония"),
    ("cloudberry", "морошка"),
    ("cornelian cherry", "кизил"),
    ("cranberry", "клюква"),
    ("crowberry", "водяника"),
    ("dewberry", "росяника"),
    ("elderberry", "бузина"),
    ("gooseberry", "крыжовник"),
    ("grape", "виноград"),
    ("guavaberry", "гуаваберри"),
    ("hackberry", "каркас"),
    ("honeyberry", "жимолость съедобная"),
    ("indian strawberry", "индийская земляника"),
    ("jamaica cherry", "ямайская вишня"),
    ("juniper berry", "можжевеловая ягода"),
    ("lingonberry", "брусника"),
    ("mock strawberry", "ложная земляника"),
    ("nannyberry", "калина гордовина"),
    ("native cherry", "австралийская вишня"),
    ("native gooseberry", "австралийский крыжовник"),
    ("pineberry", "ананасная земляника"),
    ("purple apple berry", "пурпурная яблочная ягода"),
    ("raspberry", "малина"),
    ("riberry", "риберри"),
    ("snowberry", "снежноягодник"),
    ("strawberry", "клубника"),
    ("strawberry guava", "клубничная гуава"),
    ("surinam cherry", "суринамская вишня"),
    ("tayberry", "тейберри"),
    ("thimbleberry", "малиноклён"),
    ("white mulberry", "белая шелковица"),
    ("wineberry", "винная ягода")
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Получен запрос /predict с файлом: {file.filename}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info("Изображение успешно прочитано и преобразовано в RGB")

        input_tensor = transform(image).unsqueeze(0).numpy()
        logger.debug(f"Tensor shape: {input_tensor.shape}")

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})

        pred_class = int(np.argmax(outputs[0]))
        pred_name = class_names[pred_class]

        # Сохранение картинки и предсказания в базу данных
        save_prediction(file.filename, contents, pred_name[0])

        logger.info(f"Предсказание: {pred_class} -> {pred_name}")

        # return JSONResponse(
        #     content={
        #         "predicted_class": pred_class,
        #         "predicted_name_en": pred_name[0],
        #         "predicted_name_ru": pred_name[1]
        #     }
        # )

        return {
            "predicted_class": pred_class,
            "predicted_name_en": pred_name[0],
            "predicted_name_ru": pred_name[1]
        }


    except Exception as e:
        logger.exception(f"Ошибка при обработке изображения: {e}")
        return {"error": "Ошибка при обработке изображения."}


# from fastapi import FastAPI, UploadFile, File
# from PIL import Image
# import onnxruntime as ort
# import numpy as np
# from torchvision import transforms
# import io
# import logging.config
# import yaml
# import os
#
# app = FastAPI()
#
# # Логгирование
# config_path = os.path.join(os.path.dirname(__file__), "..", "..", "logging_config.yaml")
# with open(config_path, 'r') as f:
#     config = yaml.safe_load(f)
# logging.config.dictConfig(config)
#
# # === Модель ===
# base_dir = os.path.dirname(__file__)
# ONNX_MODEL_PATH = os.path.join(base_dir, "..", "..", "models", "efficientnetB1_1.0_best_model.onnx")
# session = ort.InferenceSession(ONNX_MODEL_PATH)
#
# # === Трансформ ===
# transform = transforms.Compose([
#     transforms.Resize((240, 240)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])
#
# class_names = [
#     ("barberry", "барбарис"),
#     ("bayberry", "восковница"),
#     ("bearberry", "толокнянка"),
#     ("black berry", "ежевика"),
#     ("black cherry", "черная вишня"),
#     ("blueberry", "голубика"),
#     ("buffaloberry", "буффало-ягода"),
#     ("burmese grape", "бирманский виноград"),
#     ("cape gooseberry", "перуанская физалис"),
#     ("cedar bay cherry", "кедровая вишня"),
#     ("ceylon gooseberry", "цейлонский крыжовник"),
#     ("cherry", "вишня"),
#     ("chokeberry", "арония"),
#     ("cloudberry", "морошка"),
#     ("cornelian cherry", "кизил"),
#     ("cranberry", "клюква"),
#     ("crowberry", "водяника"),
#     ("dewberry", "росяника"),
#     ("elderberry", "бузина"),
#     ("gooseberry", "крыжовник"),
#     ("grape", "виноград"),
#     ("guavaberry", "гуаваберри"),
#     ("hackberry", "каркас"),
#     ("honeyberry", "жимолость съедобная"),
#     ("indian strawberry", "индийская земляника"),
#     ("jamaica cherry", "ямайская вишня"),
#     ("juniper berry", "можжевеловая ягода"),
#     ("lingonberry", "брусника"),
#     ("mock strawberry", "ложная земляника"),
#     ("nannyberry", "калина гордовина"),
#     ("native cherry", "австралийская вишня"),
#     ("native gooseberry", "австралийский крыжовник"),
#     ("pineberry", "ананасная земляника"),
#     ("purple apple berry", "пурпурная яблочная ягода"),
#     ("raspberry", "малина"),
#     ("riberry", "риберри"),
#     ("snowberry", "снежноягодник"),
#     ("strawberry", "клубника"),
#     ("strawberry guava", "клубничная гуава"),
#     ("surinam cherry", "суринамская вишня"),
#     ("tayberry", "тейберри"),
#     ("thimbleberry", "малиноклён"),
#     ("white mulberry", "белая шелковица"),
#     ("wineberry", "винная ягода")
# ]
#
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents)).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0).numpy()
#
#     outputs = session.run(None, {"input": input_tensor})
#     pred_class = int(np.argmax(outputs[0]))
#     pred_name = class_names[pred_class]
#
#     return {
#         "predicted_class": pred_class,
#         "predicted_name": pred_name
#     }
