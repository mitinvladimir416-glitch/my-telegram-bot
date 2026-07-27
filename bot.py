import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from aiogram.filters import CommandStart
from groq import Groq, APIConnectionError, APIStatusError, RateLimitError
from dotenv import load_dotenv

# Загружаем ключи из файла .env (он должен лежать в той же папке)
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TELEGRAM_TOKEN or not GROQ_API_KEY:
    raise ValueError(
        "Не найдены ключи! Проверь, что файл .env существует и содержит "
        "TELEGRAM_TOKEN и GROQ_API_KEY"
    )

MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.6-27b"
HISTORY_FILE = "history.json"

# Лимит: не больше RATE_LIMIT_COUNT запросов за RATE_LIMIT_WINDOW секунд на пользователя
RATE_LIMIT_COUNT = 5
RATE_LIMIT_WINDOW = 60

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
# max_retries=0 — чтобы при лимите Groq бот сразу отвечал понятным сообщением,
# а не "молчал" по 20-30 секунд, пока SDK сам пытается повторить запрос
groq_client = Groq(api_key=GROQ_API_KEY, max_retries=0)

SYSTEM_PROMPT = "Ты дружелюбный ассистент, отвечай кратко и по делу на русском языке."

WELCOME_IMAGE = "welcome.jpg"
WELCOME_TEXT = (
    "🤖 <b>ЙО! ДОБРО ПОЖАЛОВАТЬ В МИР БОТЯРЫ!</b>\n\n"
    "Ты нажал /start — а значит, обратной дороги нет 😈\n\n"
    "Тут можно:\n"
    "💬 Просто общаться — отвечу на любой вопрос\n"
    "🎤 Скидывать войсы — слушаю внимательно\n"
    "📸 Кидать фотки — разберу, что на них\n\n"
    "Погнали, я на связи 24/7 🔥\n"
    "/help — если заблудился"
)


def load_histories() -> dict:
    """Загружает историю диалогов из файла при запуске бота."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                # Ключи в JSON всегда строки, а user_id — число, поэтому конвертируем обратно
                raw = json.load(f)
                return {int(k): v for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError):
            logging.warning("Не удалось прочитать history.json, начинаем с чистой истории")
    return {}


def save_histories():
    """Сохраняет текущую историю диалогов в файл."""
    try:
        full_path = os.path.abspath(HISTORY_FILE)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(user_histories, f, ensure_ascii=False, indent=2)
        logging.info(f"История сохранена в: {full_path}")
    except Exception as e:
        logging.exception(f"НЕ УДАЛОСЬ сохранить историю: {e}")


# Память диалога для каждого пользователя — теперь подгружается из файла при старте
user_histories = load_histories()


def clean_reply(text: str) -> str:
    """Убирает следы внутренних рассуждений модели, если они просочились в ответ."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# Храним время последних запросов каждого пользователя для лимита
user_request_times: dict[int, list[float]] = {}


def is_rate_limited(user_id: int) -> bool:
    """Возвращает True, если пользователь превысил лимит запросов."""
    now = time.time()
    timestamps = user_request_times.setdefault(user_id, [])
    # Оставляем только те запросы, что были в пределах окна
    recent = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    user_request_times[user_id] = recent

    if len(recent) >= RATE_LIMIT_COUNT:
        return True

    recent.append(now)
    return False


def describe_groq_error(e: Exception) -> str:
    """Превращает техническую ошибку Groq в понятное пользователю сообщение."""
    if isinstance(e, RateLimitError):
        return "Сейчас слишком много запросов к нейросети. Подожди немного и попробуй снова."
    if isinstance(e, APIConnectionError):
        return "Не получилось связаться с нейросетью — проблема с сетью. Попробуй ещё раз через минуту."
    if isinstance(e, APIStatusError):
        return f"Нейросеть вернула ошибку (код {e.status_code}). Попробуй чуть позже."
    return "Произошла непредвиденная ошибка. Попробуй ещё раз."


@dp.message(CommandStart())
async def cmd_start(message: Message):
    user_histories[message.from_user.id] = []

    if os.path.exists(WELCOME_IMAGE):
        photo = FSInputFile(WELCOME_IMAGE)
        await message.answer_photo(photo, caption=WELCOME_TEXT, parse_mode="HTML")
    else:
        # Если картинка не найдена — просто отправляем текст, чтобы бот не падал
        logging.warning(f"Файл {WELCOME_IMAGE} не найден, отправляю только текст")
        await message.answer(WELCOME_TEXT, parse_mode="HTML")


@dp.message(F.text == "/reset")
async def cmd_reset(message: Message):
    user_histories[message.from_user.id] = []
    save_histories()
    await message.answer("История диалога очищена.")


@dp.message(F.text == "/help")
async def cmd_help(message: Message):
    await message.answer(
        "Что я умею:\n"
        "— Просто пиши мне вопросы, и я отвечу с помощью нейросети\n"
        "— /reset — очистить историю нашего диалога\n"
        "— /help — показать это сообщение"
    )


async def get_ai_reply(user_id: int, user_text: str) -> str:
    """Отправляет текст в Groq и возвращает ответ, обновляя историю диалога."""
    history = user_histories.setdefault(user_id, [])
    history.append({"role": "user", "content": user_text})
    trimmed_history = history[-20:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed_history

    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        reply = clean_reply(response.choices[0].message.content)
    except Exception as e:
        logging.exception("Ошибка при обращении к Groq API")
        reply = describe_groq_error(e)

    history.append({"role": "assistant", "content": reply})
    save_histories()
    return reply


@dp.message(F.text)
async def handle_message(message: Message):
    if is_rate_limited(message.from_user.id):
        await message.answer(
            f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
            f"запросов в {RATE_LIMIT_WINDOW} секунд)."
        )
        return

    await bot.send_chat_action(message.chat.id, "typing")
    reply = await get_ai_reply(message.from_user.id, message.text)
    await message.answer(reply)


@dp.message(F.voice)
async def handle_voice(message: Message):
    if is_rate_limited(message.from_user.id):
        await message.answer(
            f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
            f"запросов в {RATE_LIMIT_WINDOW} секунд)."
        )
        return

    await bot.send_chat_action(message.chat.id, "typing")

    # Скачиваем голосовое сообщение во временный файл
    voice_file = await bot.get_file(message.voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
    await bot.download_file(voice_file.file_path, destination=tmp_path)

    try:
        # Распознаём речь через Groq Whisper
        with open(tmp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language="ru",
            )
        recognized_text = transcription.text
    except Exception as e:
        logging.exception("Ошибка распознавания голоса")
        await message.answer(f"Не удалось распознать голосовое сообщение: {e}")
        return
    finally:
        os.remove(tmp_path)

    if not recognized_text.strip():
        await message.answer("Не удалось разобрать речь, попробуй ещё раз.")
        return

    reply = await get_ai_reply(message.from_user.id, recognized_text)
    await message.answer(f"🎤 Я услышал: «{recognized_text}»\n\n{reply}")


@dp.message(F.photo)
async def handle_photo(message: Message):
    if is_rate_limited(message.from_user.id):
        await message.answer(
            f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
            f"запросов в {RATE_LIMIT_WINDOW} секунд)."
        )
        return

    await bot.send_chat_action(message.chat.id, "typing")

    # Берём самую качественную версию фото (последняя в списке)
    photo = message.photo[-1]
    photo_file = await bot.get_file(photo.file_id)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    await bot.download_file(photo_file.file_path, destination=tmp_path)

    try:
        with open(tmp_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    finally:
        os.remove(tmp_path)

    # Если есть подпись к фото — используем её как вопрос, иначе просим просто описать
    question = message.caption if message.caption else "Опиши, что изображено на этой картинке."

    try:
        response = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ],
            temperature=0.7,
            max_tokens=2048,
            reasoning_format="hidden",
        )
        reply = clean_reply(response.choices[0].message.content)
    except Exception as e:
        logging.exception("Ошибка анализа изображения")
        reply = describe_groq_error(e)

    # Добавляем в историю как текстовый обмен, чтобы бот помнил контекст
    user_id = message.from_user.id
    history = user_histories.setdefault(user_id, [])
    history.append({"role": "user", "content": f"[Отправил фото] {question}"})
    history.append({"role": "assistant", "content": reply})
    save_histories()

    await message.answer(reply)


async def main():
    print("Бот запущен. Не закрывай это окно.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
