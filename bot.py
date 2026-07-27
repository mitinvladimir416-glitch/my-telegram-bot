import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    FSInputFile,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.filters import CommandStart
from groq import Groq, APIConnectionError, APIStatusError, RateLimitError
from dotenv import load_dotenv

# Загружаем ключи из файла .env (он должен лежать в той же папке)
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ADMIN_ID = os.getenv("ADMIN_ID")  # твой Telegram ID — только ты сможешь делать рассылку

if not TELEGRAM_TOKEN or not GROQ_API_KEY:
    raise ValueError(
        "Не найдены ключи! Проверь, что файл .env существует и содержит "
        "TELEGRAM_TOKEN и GROQ_API_KEY"
    )

ADMIN_ID = int(ADMIN_ID) if ADMIN_ID else None

MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.6-27b"
HISTORY_FILE = "history.json"
USERS_FILE = "users.json"

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
    "/help — если заблудился\n"
    "/menu — открыть меню с кнопками"
)


def load_json_file(path: str, default):
    """Универсальная загрузка JSON-файла с обработкой ошибок."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            logging.warning(f"Не удалось прочитать {path}, начинаем с пустого значения")
    return default


def load_histories() -> dict:
    """Загружает историю диалогов из файла при запуске бота."""
    raw = load_json_file(HISTORY_FILE, {})
    # Ключи в JSON всегда строки, а user_id — число, поэтому конвертируем обратно
    return {int(k): v for k, v in raw.items()}


def save_histories():
    """Сохраняет текущую историю диалогов в файл."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(user_histories, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.exception(f"НЕ УДАЛОСЬ сохранить историю: {e}")


def load_users() -> set:
    """Загружает список ID всех, кто когда-либо писал боту (для рассылки)."""
    raw = load_json_file(USERS_FILE, [])
    return set(raw)


def save_users():
    """Сохраняет список пользователей в файл."""
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(registered_users), f)
    except Exception as e:
        logging.exception(f"НЕ УДАЛОСЬ сохранить список пользователей: {e}")


def register_user(user_id: int):
    """Запоминает пользователя, чтобы потом можно было сделать ему рассылку."""
    if user_id not in registered_users:
        registered_users.add(user_id)
        save_users()


# Память диалога и список пользователей — подгружаются из файлов при старте
user_histories = load_histories()
registered_users = load_users()

# Подтягиваем в список рассылки всех, кто уже есть в истории переписки
# (важно после обновления бота, когда users.json ещё не существовал)
_users_before = len(registered_users)
registered_users |= set(user_histories.keys())
if len(registered_users) != _users_before:
    save_users()


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


# ==================== МЕНЮ С КНОПКАМИ ====================

def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="💬 Общение", callback_data="menu_chat")],
            [InlineKeyboardButton(text="🌐 Переводчик", callback_data="menu_translate")],
            [InlineKeyboardButton(text="ℹ️ Помощь", callback_data="menu_help")],
        ]
    )


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")]]
    )


MENU_MAIN_TEXT = "📋 <b>Главное меню</b>\n\nВыбери, что хочешь сделать:"
MENU_CHAT_TEXT = (
    "💬 <b>Режим общения</b>\n\n"
    "Просто пиши мне сообщения текстом, голосом или фото — я отвечу с помощью нейросети.\n"
    "Всё уже работает, ничего дополнительно нажимать не нужно 🙂"
)
MENU_TRANSLATE_TEXT = (
    "🌐 <b>Переводчик</b>\n\n"
    "Используй команду:\n"
    "<code>/translate текст</code> — авто-перевод (RU→EN, другое→RU)\n"
    "<code>/translate en текст</code> — перевод на конкретный язык\n\n"
    "Например: <code>/translate en Привет, как дела?</code>"
)
MENU_HELP_TEXT = (
    "ℹ️ <b>Помощь</b>\n\n"
    "— Просто пиши мне вопросы, и я отвечу с помощью нейросети\n"
    "— /reset — очистить историю нашего диалога\n"
    "— /translate — перевод текста\n"
    "— /menu — открыть это меню\n"
    "— /help — показать список команд"
)


@dp.message(F.text == "/menu")
async def cmd_menu(message: Message):
    await message.answer(MENU_MAIN_TEXT, reply_markup=main_menu_keyboard(), parse_mode="HTML")


@dp.callback_query(F.data == "menu_chat")
async def callback_menu_chat(callback: CallbackQuery):
    await callback.message.edit_text(MENU_CHAT_TEXT, reply_markup=back_keyboard(), parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "menu_translate")
async def callback_menu_translate(callback: CallbackQuery):
    await callback.message.edit_text(MENU_TRANSLATE_TEXT, reply_markup=back_keyboard(), parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "menu_help")
async def callback_menu_help(callback: CallbackQuery):
    await callback.message.edit_text(MENU_HELP_TEXT, reply_markup=back_keyboard(), parse_mode="HTML")
    await callback.answer()


@dp.callback_query(F.data == "menu_back")
async def callback_menu_back(callback: CallbackQuery):
    await callback.message.edit_text(MENU_MAIN_TEXT, reply_markup=main_menu_keyboard(), parse_mode="HTML")
    await callback.answer()


# ==================== ОСНОВНЫЕ КОМАНДЫ ====================

@dp.message(CommandStart())
async def cmd_start(message: Message):
    register_user(message.from_user.id)
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
    await message.answer(MENU_HELP_TEXT, parse_mode="HTML")


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


async def translate_text(text: str, target_lang: str | None = None) -> str:
    """Переводит текст через нейросеть. Если target_lang не задан — авто: RU->EN, иначе->RU."""
    if target_lang:
        instruction = (
            f"Переведи следующий текст на язык с кодом '{target_lang}'. "
            "Ответь ТОЛЬКО переводом, без пояснений, кавычек и комментариев."
        )
    else:
        instruction = (
            "Определи язык текста. Если текст на русском — переведи на английский. "
            "Если текст на любом другом языке — переведи на русский. "
            "Сохрани тон и смысл максимально точно. "
            "Ответь ТОЛЬКО переводом, без пояснений, кавычек, комментариев и указания языка."
        )

    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return clean_reply(response.choices[0].message.content)
    except Exception as e:
        logging.exception("Ошибка при переводе")
        return describe_groq_error(e)


@dp.message(F.text.startswith("/translate"))
async def cmd_translate(message: Message):
    register_user(message.from_user.id)

    if is_rate_limited(message.from_user.id):
        await message.answer(
            f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
            f"запросов в {RATE_LIMIT_WINDOW} секунд)."
        )
        return

    # Убираем саму команду "/translate" из текста
    remainder = message.text[len("/translate"):].strip()

    if not remainder:
        await message.answer(
            "Использование:\n"
            "/translate <текст> — авто-перевод (RU→EN, другое→RU)\n"
            "/translate <код языка> <текст> — перевод на конкретный язык\n\n"
            "Например: /translate en Привет, как дела?"
        )
        return

    # Проверяем, не начинается ли текст с короткого кода языка (2-3 буквы + пробел)
    parts = remainder.split(maxsplit=1)
    target_lang = None
    text_to_translate = remainder

    if len(parts) == 2 and re.fullmatch(r"[a-zA-Z]{2,3}", parts[0]):
        target_lang = parts[0].lower()
        text_to_translate = parts[1]

    await bot.send_chat_action(message.chat.id, "typing")
    translation = await translate_text(text_to_translate, target_lang)
    await message.answer(f"🌐 {translation}")


# ==================== РАССЫЛКА (ТОЛЬКО ДЛЯ АДМИНА) ====================

def is_admin(user_id: int) -> bool:
    return ADMIN_ID is not None and user_id == ADMIN_ID


@dp.message(F.photo & F.caption.startswith("/broadcast"))
async def cmd_broadcast_photo(message: Message):
    """Рассылка картинки с подписью всем пользователям. Формат: /broadcast текст (как подпись к фото)."""
    if not is_admin(message.from_user.id):
        return  # Не админ — молча игнорируем, чтобы не подсказывать команду чужим

    text = message.caption[len("/broadcast"):].strip()
    photo_id = message.photo[-1].file_id

    await message.answer(f"Начинаю рассылку с картинкой на {len(registered_users)} пользователей...")

    sent = 0
    failed = 0
    for user_id in list(registered_users):
        try:
            await bot.send_photo(user_id, photo_id, caption=text if text else None)
            sent += 1
        except Exception as e:
            logging.warning(f"Не удалось отправить рассылку пользователю {user_id}: {e}")
            failed += 1
        await asyncio.sleep(0.05)  # небольшая пауза, чтобы не упереться в лимиты Telegram

    await message.answer(f"Готово! Успешно: {sent}, не удалось: {failed}")


@dp.message(F.text.startswith("/broadcast"))
async def cmd_broadcast_text(message: Message):
    """Текстовая рассылка всем пользователям. Формат: /broadcast текст сообщения."""
    if not is_admin(message.from_user.id):
        return  # Не админ — молча игнорируем

    text = message.text[len("/broadcast"):].strip()

    if not text:
        await message.answer(
            "Использование:\n"
            "/broadcast текст — разослать текст всем пользователям\n"
            "Или прикрепи картинку с подписью, начинающейся на /broadcast, чтобы разослать с картинкой"
        )
        return

    await message.answer(f"Начинаю рассылку на {len(registered_users)} пользователей...")

    sent = 0
    failed = 0
    for user_id in list(registered_users):
        try:
            await bot.send_message(user_id, text)
            sent += 1
        except Exception as e:
            logging.warning(f"Не удалось отправить рассылку пользователю {user_id}: {e}")
            failed += 1
        await asyncio.sleep(0.05)

    await message.answer(f"Готово! Успешно: {sent}, не удалось: {failed}")


# ==================== ОБЫЧНЫЕ СООБЩЕНИЯ ====================

@dp.message(F.text)
async def handle_message(message: Message):
    register_user(message.from_user.id)

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
    register_user(message.from_user.id)

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
    register_user(message.from_user.id)

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
