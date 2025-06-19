import asyncio
import logging
import os
import yaml
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from handlers import router
import logging.config

# Настройка логирования
config_path = os.path.join(os.path.dirname(__file__), "..", "..", "logging_config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
logging.config.dictConfig(config)

logger = logging.getLogger("bot")

# Токен бота
BOT_TOKEN = '8098658423:AAE8jPQr2DVdtUL_PB5Ez0UmtbIGyQlCnEc'


async def main():
    try:
        logger.info("Инициализация Telegram-бота...")

        bot = Bot(
            token=BOT_TOKEN,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        logger.info("Бот успешно создан.")

        dp = Dispatcher()
        dp.include_router(router)
        logger.info("Роутеры подключены.")

        logger.info("Запуск polling...")
        await dp.start_polling(bot)
    except Exception as e:
        logger.exception(f"Ошибка при запуске бота: {e}")
    finally:
        logger.info("Завершение работы бота.")

if __name__ == "__main__":
    asyncio.run(main())


# import asyncio
# import logging
# from aiogram import Bot, Dispatcher
# from aiogram.enums import ParseMode
# from aiogram.client.default import DefaultBotProperties
# from handlers import router
# import logging.config
# import yaml
# import os
#
# # Логгирование
# config_path = os.path.join(os.path.dirname(__file__), "..", "..", "logging_config.yaml")
# with open(config_path, 'r') as f:
#     config = yaml.safe_load(f)
# logging.config.dictConfig(config)
#
#
# BOT_TOKEN = '8098658423:AAE8jPQr2DVdtUL_PB5Ez0UmtbIGyQlCnEc'
#
# async def main():
#     logging.basicConfig(level=logging.INFO)
#
#     bot = Bot(
#         token=BOT_TOKEN,
#         default=DefaultBotProperties(parse_mode=ParseMode.HTML)
#     )
#
#     dp = Dispatcher()
#     dp.include_router(router)  # 👈 подключаем router из handlers.py
#
#     await dp.start_polling(bot)
#
# if __name__ == "__main__":
#     asyncio.run(main())
