import logging
from aiogram import Router, F
from aiogram.types import Message
import requests
from aiogram import Bot
import aiohttp

# === Логгер ===
logger = logging.getLogger("bot.handlers")

# === Роутер ===
router = Router()

API_URL = "http://localhost:8000/predict"  # URL FastAPI

@router.message(F.text == "/start")
async def handle_start(message: Message):
    logger.info(f"Получена команда /start от пользователя {message.from_user.id}")
    await message.answer("Привет! Отправь фото ягоды, и я попробую её распознать.")

@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot):
    logger.info(f"Получено фото от пользователя {message.from_user.id}")
    await message.answer("Фото получено, анализирую...")

    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file_path = file_info.file_path
    file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_path}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                if resp.status == 200:
                    image_bytes = await resp.read()
                    logger.info(f"Фото успешно загружено с Telegram для пользователя {message.from_user.id}")

                    files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
                    try:
                        response = requests.post(API_URL, files=files)
                        response.encoding = 'utf-8'
                        if response.status_code == 200:
                            print(response.text)
                            data = response.json()
                            name_en = data.get("predicted_name_en", "?")
                            name_ru = data.get("predicted_name_ru", "?")
                            logger.info(f"Предсказание отправлено пользователю {message.from_user.id}: {name_ru} / {name_en}")
                            await message.answer(
                                f"Предсказание:\n"
                                f"🇷🇺 <b>{name_ru}</b>\n"
                                f"🇬🇧 <i>{name_en}</i>\n"
                            )
                        else:
                            logger.warning(f"Ошибка от API {API_URL}: статус {response.status_code}")
                            await message.answer(f"Ошибка сервера: {response.status_code}")
                    except Exception as e:
                        logger.exception(f"Ошибка при обращении к API: {e}")
                        await message.answer(f"Ошибка API:\n<code>{e}</code>")
                else:
                    logger.error(f"Ошибка загрузки фото с Telegram. Статус: {resp.status}")
                    await message.answer("Не удалось загрузить изображение.")
    except Exception as e:
        logger.exception(f"Ошибка при получении фото от Telegram: {e}")
        await message.answer("Произошла ошибка при обработке изображения.")


# from aiogram import Router, F
# from aiogram.types import Message
# import requests
#
# router = Router()
#
# API_URL = "http://localhost:8000/predict"  # URL FastAPI
#
# @router.message(F.text == "/start")
# async def handle_start(message: Message):
#     await message.answer("Привет! Отправь фото ягоды, и я попробую её распознать. 🍓")
#
# from aiogram import Bot
# import aiohttp
#
# @router.message(F.photo)
# async def handle_photo(message: Message, bot: Bot):
#     await message.answer("📥 Фото получено, загружаю...")
#
#     photo = message.photo[-1]  # Самое качественное изображение
#     file_info = await bot.get_file(photo.file_id)
#     file_path = file_info.file_path
#
#     # Получаем файл через HTTP
#     file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_path}"
#
#     async with aiohttp.ClientSession() as session:
#         async with session.get(file_url) as resp:
#             if resp.status == 200:
#                 image_bytes = await resp.read()
#
#                 # Отправляем на API
#                 files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
#                 try:
#                     response = requests.post(API_URL, files=files)
#                     if response.status_code == 200:
#                         data = response.json()
#                         name_en, name_ru = data.get("predicted_name", ["?", "?"])
#                         await message.answer(
#                             f"Предсказание:\n"
#                             f"🇷🇺 <b>{name_ru}</b>\n"
#                             f"🇬🇧 <i>{name_en}</i>\n"
#                         )
#                     else:
#                         await message.answer(f"Ошибка сервера: {response.status_code}")
#                 except Exception as e:
#                     await message.answer(f"Ошибка API:\n<code>{e}</code>")
#             else:
#                 await message.answer("Не удалось загрузить изображение.")
