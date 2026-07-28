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
PROMPT_STICKER_ID = os.getenv("PROMPT_STICKER_ID")  # file_id стикера, который показывается перед готовым промптом

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


async def _typing_loop(chat_id: int):
    """Периодически шлёт 'печатает...', пока идёт долгий запрос к нейросети —
    иначе Telegram гасит индикатор через ~5 секунд."""
    try:
        while True:
            await bot.send_chat_action(chat_id, "typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


async def with_typing(chat_id: int, coro):
    """Оборачивает долгий вызов нейросети, поддерживая индикатор 'печатает...' всё время."""
    typing_task = asyncio.create_task(_typing_loop(chat_id))
    try:
        return await coro
    finally:
        typing_task.cancel()


async def send_prompt_reply(chat_id: int, reply_text: str, keyboard: InlineKeyboardMarkup):
    """Отправляет ответ из раздела 'Промпты'. Если это финальный готовый промпт —
    сначала на 3-4 секунды показывает стикер (для эффектности), потом удаляет его."""
    if "ГОТОВЫЙ ПРОМПТ:" in reply_text and PROMPT_STICKER_ID:
        try:
            sticker_message = await bot.send_sticker(chat_id, PROMPT_STICKER_ID)
            await asyncio.sleep(3.5)
            try:
                await bot.delete_message(chat_id, sticker_message.message_id)
            except Exception:
                logging.exception("Не удалось удалить стикер после показа")
        except Exception:
            logging.exception("Не удалось отправить стикер перед промптом")

    await bot.send_message(chat_id, reply_text, reply_markup=keyboard)

SYSTEM_PROMPT = "Ты дружелюбный ассистент, отвечай кратко и по делу на русском языке."

WELCOME_IMAGE = "welcome.jpg"
WELCOME_TEXT = (
    "🤖 <b>ЙО! ДОБРО ПОЖАЛОВАТЬ В МИР БОТЯРЫ!</b>\n\n"
    "Нажал старт — красавчик, теперь ты в деле 😈\n\n"
    "Что я умею:\n"
    "💬 Шарю за любые темы — просто пиши\n"
    "🎤 Понимаю войсы — шли голосом\n"
    "📸 Разбираю фотки — гружу, вижу, отвечаю\n"
    "🌐 Шпарю переводы на любой язык\n"
    "🎨 Собираю крутые промпты — для музыки в Suno, картинок и видео. "
    "Скажи, что хочешь получить, и я выкачу готовый промпт под нужную нейронку\n\n"
    "Погнали, я на связи 24/7 🔥\n"
    "Выбирай режим прямо в меню ниже 👇"
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


async def call_prompt_model(messages: list) -> str:
    """Вызов нейросети для раздела 'Промпты' (Groq)."""
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
        )
        return clean_reply(response.choices[0].message.content)
    except Exception as e:
        logging.exception("Ошибка при обращении к Groq API")
        return describe_groq_error(e)


# ==================== МЕНЮ С КНОПКАМИ ====================

# Храним, у кого сейчас включён режим перевода в чате (и на какой язык)
user_translate_mode: dict[int, dict] = {}

# Храним, у кого сейчас включён режим составления промптов (и историю диалога по теме)
user_prompt_mode: dict[int, str] = {}  # user_id -> "suno" / "image" / "video"
user_prompt_histories: dict[int, list] = {}
user_prompt_target: dict[int, str] = {}  # user_id -> выбранная версия/нейросеть (текст для показа)

# Храним состояние мастера "Обложка трека": шаг, текст песни, фото (если есть)
user_cover_state: dict[int, dict] = {}


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="💬 Общение", callback_data="menu_chat")],
            [InlineKeyboardButton(text="🌐 Переводчик", callback_data="menu_translate")],
            [InlineKeyboardButton(text="🎨 Промпты", callback_data="menu_prompts")],
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
MENU_HELP_TEXT = (
    "ℹ️ <b>Помощь</b>\n\n"
    "— Просто пиши мне вопросы, и я отвечу с помощью нейросети\n"
    "— /reset — очистить историю нашего диалога\n"
    "— /translate — перевод текста командой\n"
    "— /menu — открыть это меню\n"
    "— /help — показать список команд"
)

# Языки с быстрым доступом из меню переводчика
QUICK_LANGUAGES = {
    "en": "английский",
    "fr": "французский",
    "de": "немецкий",
}

TRANSLATE_SUBMENU_TEXT = "🌐 <b>Переводчик</b>\n\nВыбери язык — и просто присылай текст, буду переводить:"


def translate_submenu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🇬🇧 Английский", callback_data="tr_lang_en"),
                InlineKeyboardButton(text="🇫🇷 Французский", callback_data="tr_lang_fr"),
            ],
            [
                InlineKeyboardButton(text="🇩🇪 Немецкий", callback_data="tr_lang_de"),
                InlineKeyboardButton(text="🔍 Определить язык", callback_data="tr_lang_auto"),
            ],
            [InlineKeyboardButton(text="⌨️ Указать язык", callback_data="tr_lang_custom")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
        ]
    )


def translate_active_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Сменить язык", callback_data="menu_translate")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
        ]
    )


# ==================== РАЗДЕЛ "ПРОМПТЫ" ====================

PROMPTS_SUBMENU_TEXT = "🎨 <b>Помощник по промптам</b>\n\nВыбери направление:"

PROMPT_CONFIG = {
    "suno": {
        "label": "🎵 Suno (музыка)",
        "target_question": "🎵 <b>Suno</b>\n\nКакая версия?",
        "targets": [
            ("4", "Suno 4"),
            ("4.5", "Suno 4.5"),
            ("5", "Suno 5"),
            ("5.5", "Suno 5.5"),
        ],
        "intro_after_target": (
            "✅ Версия: <b>{target}</b>\n\n"
            "Теперь расскажи идею трека — жанр, настроение, тематику, есть ли вокал."
        ),
        "system_prompt": (
            "Ты — опытный саунд-продюсер и эксперт по составлению промптов для Suno AI "
            "(нейросеть для генерации музыки). Твоя задача — помочь пользователю составить "
            "качественный промпт. Веди диалог по существу:\n"
            "1. Уточни жанр/стиль, настроение, темп, наличие и пол вокала, референсы-исполнителей, "
            "структуру трека (куплет/припев/бридж), нужен ли текст песни (лирика).\n"
            "Задавай не больше 1-2 уточняющих вопросов за раз, не заваливай пользователя вопросами сразу.\n"
            "Когда данных достаточно — сформируй готовый промпт для Suno (структурированный, с тегами, "
            "если версия их поддерживает) и обязательно начни этот момент строкой 'ГОТОВЫЙ ПРОМПТ:' "
            "на отдельной строке, дальше сам промпт. Отвечай на русском, но сам текст промпта пиши "
            "так, как эффективнее для Suno (обычно это английский для тегов стиля)."
        ),
    },
    "image": {
        "label": "🖼 Картинка",
        "target_question": "🖼 <b>Картинка</b>\n\nДля какой нейросети?",
        "targets": [
            ("mj", "Midjourney"),
            ("dalle", "DALL-E 3"),
            ("sd", "Stable Diffusion"),
            ("flux", "Flux"),
        ],
        "intro_after_target": (
            "✅ Нейросеть: <b>{target}</b>\n\n"
            "Опиши идею — что должно быть на картинке.\n\n"
            "💡 Можно и по-другому: пришли фото и подписью укажи, что в нём нужно поменять — "
            "составлю промпт под правку именно этого изображения."
        ),
        "system_prompt": (
            "Ты — эксперт по составлению промптов для генерации изображений через нейросети. "
            "Твоя задача — помочь пользователю составить качественный промпт. Веди диалог по существу:\n"
            "1. Уточни сюжет и объект, стиль (фотореализм, аниме, живопись, 3D и т.д.), композицию, "
            "освещение, цветовую палитру, ракурс, соотношение сторон, дополнительные детали и настроение.\n"
            "Задавай не больше 1-2 уточняющих вопросов за раз.\n"
            "Когда данных достаточно — сформируй готовый промпт, оформленный по стандартам именно "
            "выбранной нейросети (для Midjourney — с параметрами --ar --v --style и т.д., если уместно). "
            "Обязательно начни этот момент строкой 'ГОТОВЫЙ ПРОМПТ:' на отдельной строке, дальше сам "
            "промпт. Отвечай на русском, промпт можно писать на английском, если так эффективнее."
        ),
    },
    "video": {
        "label": "🎬 Видео",
        "target_question": "🎬 <b>Видео</b>\n\nДля какой нейросети?",
        "targets": [
            ("sora", "Sora"),
            ("runway", "Runway"),
            ("kling", "Kling"),
            ("veo", "Veo"),
        ],
        "intro_after_target": (
            "✅ Нейросеть: <b>{target}</b>\n\nОпиши идею сцены."
        ),
        "system_prompt": (
            "Ты — эксперт по составлению промптов для генерации видео через нейросети "
            "(Sora, Runway, Kling, Veo, Pika и подобные). Твоя задача — помочь пользователю "
            "составить качественный промпт. Веди диалог по существу:\n"
            "1. Уточни сюжет сцены, движение камеры (панорама, наезд, статика и т.д.), стиль "
            "(кино, реализм, анимация), освещение, длительность, темп действия, звук/музыку если поддерживается.\n"
            "Задавай не больше 1-2 уточняющих вопросов за раз.\n"
            "Когда данных достаточно — сформируй готовый промпт под конкретную нейросеть. "
            "Обязательно начни этот момент строкой 'ГОТОВЫЙ ПРОМПТ:' на отдельной строке, дальше сам "
            "промпт. Отвечай на русском, промпт можно писать на английском, если так эффективнее."
        ),
    },
}


def prompts_submenu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=cfg["label"], callback_data=f"prompt_{key}")]
            for key, cfg in PROMPT_CONFIG.items()
        ]
        + [
            [InlineKeyboardButton(text="🖼 Обложка трека", callback_data="menu_cover")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
        ]
    )


def prompt_target_keyboard(topic: str) -> InlineKeyboardMarkup:
    """Клавиатура выбора версии/нейросети под конкретную тему."""
    targets = PROMPT_CONFIG[topic]["targets"]
    rows = []
    for i in range(0, len(targets), 2):
        row = [
            InlineKeyboardButton(text=label, callback_data=f"ptgt_{topic}_{code}")
            for code, label in targets[i : i + 2]
        ]
        rows.append(row)

    rows.append([InlineKeyboardButton(text="◀️ Назад", callback_data="menu_prompts")])

    return InlineKeyboardMarkup(inline_keyboard=rows)


def prompt_active_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Сменить тему", callback_data="menu_prompts")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
        ]
    )



async def get_image_prompt_from_photo(user_id: int, image_base64: str, desired_change: str) -> str:
    """Анализирует присланное фото и составляет промпт для его правки в генеративной нейросети."""
    config = PROMPT_CONFIG["image"]
    history = user_prompt_histories.setdefault(user_id, [])

    target = user_prompt_target.get(user_id)
    target_note = (
        f"Пользователь уже выбрал нейросеть: {target}. Готовый промпт формируй именно под неё, "
        "не спрашивай про это повторно. "
        if target
        else ""
    )

    combined_system = (
        config["system_prompt"]
        + "\n\nПользователь прислал фотографию и хочет внести в неё правки через генеративную "
        "нейросеть. " + target_note + "Сначала кратко опиши (для себя, но можно упомянуть в ответе), "
        "что видишь на фото, затем сразу выдай готовый промпт под правку этого фото с пометкой "
        "'ГОТОВЫЙ ПРОМПТ:'."
    )

    trimmed_history = history[-10:]
    messages = (
        [{"role": "system", "content": combined_system}]
        + trimmed_history
        + [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Хочу изменить в этом фото: {desired_change}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]
    )

    try:
        response = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            reasoning_format="hidden",
        )
        reply = clean_reply(response.choices[0].message.content)
    except Exception as e:
        logging.exception("Ошибка при анализе фото для промпта")
        reply = describe_groq_error(e)

    # В историю добавляем текстовый след, чтобы дальнейшие уточнения помнили контекст
    history.append({"role": "user", "content": f"[Прислал фото] Хочу изменить: {desired_change}"})
    history.append({"role": "assistant", "content": reply})
    return reply


async def get_prompt_reply(user_id: int, prompt_type: str, user_text: str) -> str:
    """Ведёт диалог по составлению промпта в выбранной теме (suno/image/video)."""
    config = PROMPT_CONFIG[prompt_type]
    history = user_prompt_histories.setdefault(user_id, [])
    history.append({"role": "user", "content": user_text})
    trimmed_history = history[-20:]

    system_content = config["system_prompt"]
    target = user_prompt_target.get(user_id)
    if target:
        system_content += (
            f"\n\nПользователь уже выбрал: {target}. Не спрашивай про версию/нейросеть повторно — "
            "сразу переходи к остальным уточняющим вопросам, а готовый промпт формируй именно под неё."
        )

    messages = [{"role": "system", "content": system_content}] + trimmed_history

    reply = await call_prompt_model(messages)

    history.append({"role": "assistant", "content": reply})
    return reply


# ==================== РАЗДЕЛ "ОБЛОЖКА ТРЕКА" ====================

COVER_INTRO_TEXT = (
    "🖼 <b>Обложка трека</b>\n\n"
    "Пришли текст песни (лирику) — на основе него подберу образы, настроение и стиль для обложки."
)

COVER_PHOTO_CHOICE_TEXT = (
    "✅ Текст песни принят.\n\n"
    "Хочешь добавить фото? Например, референс по стилю, фото исполнителя или просто вдохновляющую картинку — "
    "учту её при составлении промпта."
)

COVER_AWAITING_PHOTO_TEXT = "📸 Пришли фото — или нажми кнопку ниже, чтобы продолжить без него."

COVER_FORMAT_TEXT = "📐 Теперь выбери формат обложки:"

# Форматы: код -> (aspect ratio для промпта, подпись для кнопки)
COVER_FORMATS = {
    "11": ("1:1", "⬛ Квадрат 1:1"),
    "43": ("4:3", "🖼 Классический 4:3"),
    "169": ("16:9", "📺 Широкий 16:9"),
    "34": ("3:4", "📱 Портретный 3:4"),
    "916": ("9:16", "📲 Вертикальный 9:16"),
}

COVER_SYSTEM_PROMPT = (
    "Ты — эксперт по составлению промптов для генерации обложек музыкальных треков в модели "
    "ChatGPT Image 2 (нейросеть OpenAI для генерации изображений). Тебе присылают текст песни "
    "(лирику), возможно — референсное фото, и нужное соотношение сторон обложки. Твоя задача:\n"
    "1. Проанализируй текст песни — определи настроение, тематику, ключевые образы и символы.\n"
    "2. Если есть референсное фото — учти его стиль, цветовую палитру, композицию.\n"
    "3. Составь единый выразительный промпт для обложки: визуальная композиция, художественный стиль "
    "(фотореализм/иллюстрация/абстракция и т.д.), цветовая палитра, настроение, ключевые визуальные "
    "элементы. Обложка не должна содержать текст/буквы, если явно не указано иное.\n"
    "Сначала коротко (1-2 предложения на русском) опиши, какое настроение считал из лирики. "
    "Затем сразу выдай готовый промпт под пометкой 'ГОТОВЫЙ ПРОМПТ:' на отдельной строке — "
    "сам промпт пиши на английском (так эффективнее для генерации), обязательно укажи в конце "
    "нужный aspect ratio."
)


def cover_back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="◀️ Назад", callback_data="menu_prompts")]]
    )


def cover_photo_choice_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📸 Да, пришлю фото", callback_data="cover_photo_yes")],
            [InlineKeyboardButton(text="⏭ Без фото", callback_data="cover_photo_no")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_prompts")],
        ]
    )


def cover_awaiting_photo_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⏭ Продолжить без фото", callback_data="cover_photo_no")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_prompts")],
        ]
    )


def cover_format_keyboard() -> InlineKeyboardMarkup:
    codes = list(COVER_FORMATS.keys())
    rows = []
    for i in range(0, len(codes), 2):
        row = [
            InlineKeyboardButton(text=COVER_FORMATS[c][1], callback_data=f"cover_fmt_{c}")
            for c in codes[i : i + 2]
        ]
        rows.append(row)
    rows.append([InlineKeyboardButton(text="◀️ Назад", callback_data="menu_prompts")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def cover_result_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Новая обложка", callback_data="menu_cover")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
        ]
    )


async def generate_cover_prompt(lyrics: str, photo_base64: str | None, ratio: str) -> str:
    """Анализирует текст песни (и фото, если есть) и составляет промпт для обложки трека."""
    user_content_text = f"Текст песни:\n{lyrics}\n\nНужное соотношение сторон: {ratio}"

    if photo_base64:
        messages = [
            {"role": "system", "content": COVER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{photo_base64}"},
                    },
                ],
            },
        ]
        model = VISION_MODEL
        extra_kwargs = {"reasoning_format": "hidden"}
    else:
        messages = [
            {"role": "system", "content": COVER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content_text},
        ]
        model = MODEL
        extra_kwargs = {}

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
            **extra_kwargs,
        )
        return clean_reply(response.choices[0].message.content)
    except Exception as e:
        logging.exception("Ошибка при составлении промпта обложки")
        return describe_groq_error(e)



@dp.message(F.text == "/menu")
async def cmd_menu(message: Message):
    user_translate_mode.pop(message.from_user.id, None)
    user_prompt_mode.pop(message.from_user.id, None)
    user_prompt_target.pop(message.from_user.id, None)
    user_cover_state.pop(message.from_user.id, None)
    await message.answer(MENU_MAIN_TEXT, reply_markup=main_menu_keyboard(), parse_mode="HTML")


async def show_menu_screen(callback: CallbackQuery, text: str, keyboard: InlineKeyboardMarkup):
    """Показывает экран меню. Если исходное сообщение — фото (например, из /start),
    редактировать его текст нельзя, поэтому удаляем и присылаем новое текстовое сообщение."""
    try:
        if callback.message.photo:
            await callback.message.delete()
            await callback.message.answer(text, reply_markup=keyboard, parse_mode="HTML")
        else:
            await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    except Exception:
        logging.exception("Ошибка при обновлении экрана меню")
        try:
            await callback.message.answer(text, reply_markup=keyboard, parse_mode="HTML")
        except Exception:
            logging.exception("Не удалось отправить меню даже отдельным сообщением")
    finally:
        await callback.answer()


@dp.callback_query(F.data == "menu_chat")
async def callback_menu_chat(callback: CallbackQuery):
    user_translate_mode.pop(callback.from_user.id, None)
    user_prompt_mode.pop(callback.from_user.id, None)
    user_prompt_target.pop(callback.from_user.id, None)
    user_cover_state.pop(callback.from_user.id, None)
    await show_menu_screen(callback, MENU_CHAT_TEXT, back_keyboard())


@dp.callback_query(F.data == "menu_translate")
async def callback_menu_translate(callback: CallbackQuery):
    # Выходим из режима перевода/промптов при повторном открытии подменю
    user_translate_mode.pop(callback.from_user.id, None)
    user_prompt_mode.pop(callback.from_user.id, None)
    user_prompt_target.pop(callback.from_user.id, None)
    user_cover_state.pop(callback.from_user.id, None)
    await show_menu_screen(callback, TRANSLATE_SUBMENU_TEXT, translate_submenu_keyboard())


@dp.callback_query(F.data.in_(["tr_lang_en", "tr_lang_fr", "tr_lang_de"]))
async def callback_translate_quick_lang(callback: CallbackQuery):
    lang_code = callback.data.split("_")[-1]
    lang_name = QUICK_LANGUAGES[lang_code]
    user_translate_mode[callback.from_user.id] = {"target_lang": lang_code}
    await show_menu_screen(
        callback,
        f"✅ Режим перевода на <b>{lang_name}</b> включён.\n\nПросто присылай текст — переведу.",
        translate_active_keyboard(),
    )


@dp.callback_query(F.data == "tr_lang_auto")
async def callback_translate_auto(callback: CallbackQuery):
    user_translate_mode[callback.from_user.id] = {"target_lang": None}
    await show_menu_screen(
        callback,
        "✅ Режим <b>автоопределения</b> включён (RU→EN, другой язык→RU).\n\n"
        "Просто присылай текст — переведу.",
        translate_active_keyboard(),
    )


@dp.callback_query(F.data == "tr_lang_custom")
async def callback_translate_custom(callback: CallbackQuery):
    user_translate_mode[callback.from_user.id] = {"awaiting_custom_lang": True}
    await show_menu_screen(
        callback,
        "⌨️ Напиши, на какой язык переводить.\n"
        "Можно и код (es, ja, pt), и название (испанский, japanese).",
        translate_active_keyboard(),
    )


@dp.callback_query(F.data == "menu_help")
async def callback_menu_help(callback: CallbackQuery):
    await show_menu_screen(callback, MENU_HELP_TEXT, back_keyboard())


@dp.callback_query(F.data == "menu_prompts")
async def callback_menu_prompts(callback: CallbackQuery):
    user_prompt_mode.pop(callback.from_user.id, None)
    user_prompt_target.pop(callback.from_user.id, None)
    user_cover_state.pop(callback.from_user.id, None)
    await show_menu_screen(callback, PROMPTS_SUBMENU_TEXT, prompts_submenu_keyboard())


@dp.callback_query(F.data == "menu_cover")
async def callback_menu_cover(callback: CallbackQuery):
    user_id = callback.from_user.id
    user_prompt_mode.pop(user_id, None)
    user_prompt_target.pop(user_id, None)
    user_cover_state[user_id] = {"step": "lyrics"}
    await show_menu_screen(callback, COVER_INTRO_TEXT, cover_back_keyboard())


@dp.callback_query(F.data == "cover_photo_yes")
async def callback_cover_photo_yes(callback: CallbackQuery):
    user_id = callback.from_user.id
    state = user_cover_state.get(user_id)
    if not state:
        await callback.answer("Сессия сброшена, начни заново через меню.", show_alert=True)
        return
    state["step"] = "awaiting_photo"
    await show_menu_screen(callback, COVER_AWAITING_PHOTO_TEXT, cover_awaiting_photo_keyboard())


@dp.callback_query(F.data == "cover_photo_no")
async def callback_cover_photo_no(callback: CallbackQuery):
    user_id = callback.from_user.id
    state = user_cover_state.get(user_id)
    if not state:
        await callback.answer("Сессия сброшена, начни заново через меню.", show_alert=True)
        return
    state["step"] = "format"
    state.pop("photo_base64", None)
    await show_menu_screen(callback, COVER_FORMAT_TEXT, cover_format_keyboard())


@dp.callback_query(F.data.startswith("cover_fmt_"))
async def callback_cover_format(callback: CallbackQuery):
    user_id = callback.from_user.id
    state = user_cover_state.get(user_id)
    if not state or "lyrics" not in state:
        await callback.answer("Сессия сброшена, начни заново через меню.", show_alert=True)
        return

    code = callback.data[len("cover_fmt_"):]
    ratio, _label = COVER_FORMATS.get(code, ("1:1", ""))

    if is_rate_limited(user_id):
        await callback.answer(
            f"Слишком много запросов подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
            f"запросов в {RATE_LIMIT_WINDOW} секунд).",
            show_alert=True,
        )
        return

    await callback.answer()

    chat_id = callback.message.chat.id
    lyrics = state["lyrics"]
    photo_base64 = state.get("photo_base64")

    reply = await with_typing(chat_id, generate_cover_prompt(lyrics, photo_base64, ratio))
    user_cover_state.pop(user_id, None)
    await send_prompt_reply(chat_id, reply, cover_result_keyboard())


@dp.callback_query(F.data.in_([f"prompt_{k}" for k in PROMPT_CONFIG.keys()]))
async def callback_prompt_select(callback: CallbackQuery):
    """Пользователь выбрал тему (Suno/Картинка/Видео) — показываем выбор версии/нейросети."""
    topic = callback.data.split("_", 1)[1]
    user_id = callback.from_user.id
    user_prompt_mode[user_id] = topic
    user_prompt_target.pop(user_id, None)
    config = PROMPT_CONFIG[topic]
    await show_menu_screen(callback, config["target_question"], prompt_target_keyboard(topic))


@dp.callback_query(F.data.startswith("ptgt_"))
async def callback_prompt_target(callback: CallbackQuery):
    """Пользователь выбрал версию/нейросеть — начинаем диалог по существу."""
    _, topic, code = callback.data.split("_", 2)
    user_id = callback.from_user.id
    config = PROMPT_CONFIG[topic]

    target_label = next((label for c, label in config["targets"] if c == code), code)
    user_prompt_target[user_id] = target_label
    user_prompt_mode[user_id] = topic
    user_prompt_histories[user_id] = []  # начинаем тему с чистого листа

    intro = config["intro_after_target"].format(target=target_label)
    await show_menu_screen(callback, intro, prompt_active_keyboard())


@dp.callback_query(F.data == "menu_back")
async def callback_menu_back(callback: CallbackQuery):
    user_translate_mode.pop(callback.from_user.id, None)
    user_prompt_mode.pop(callback.from_user.id, None)
    user_prompt_target.pop(callback.from_user.id, None)
    user_cover_state.pop(callback.from_user.id, None)
    await show_menu_screen(callback, MENU_MAIN_TEXT, main_menu_keyboard())


# ==================== ОСНОВНЫЕ КОМАНДЫ ====================

@dp.message(CommandStart())
async def cmd_start(message: Message):
    register_user(message.from_user.id)
    user_histories[message.from_user.id] = []
    user_translate_mode.pop(message.from_user.id, None)
    user_prompt_mode.pop(message.from_user.id, None)
    user_prompt_target.pop(message.from_user.id, None)
    user_cover_state.pop(message.from_user.id, None)

    if os.path.exists(WELCOME_IMAGE):
        photo = FSInputFile(WELCOME_IMAGE)
        await message.answer_photo(
            photo, caption=WELCOME_TEXT, reply_markup=main_menu_keyboard(), parse_mode="HTML"
        )
    else:
        # Если картинка не найдена — просто отправляем текст, чтобы бот не падал
        logging.warning(f"Файл {WELCOME_IMAGE} не найден, отправляю только текст")
        await message.answer(WELCOME_TEXT, reply_markup=main_menu_keyboard(), parse_mode="HTML")


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
    translation = await with_typing(message.chat.id, translate_text(text_to_translate, target_lang))
    await message.answer(f"🌐 {translation}")


# ==================== РАССЫЛКА (ТОЛЬКО ДЛЯ АДМИНА) ====================

def is_admin(user_id: int) -> bool:
    return ADMIN_ID is not None and user_id == ADMIN_ID


@dp.message(F.sticker)
async def handle_sticker(message: Message):
    """Если админ присылает стикер боту — подсказываем его file_id для настройки PROMPT_STICKER_ID."""
    if is_admin(message.from_user.id):
        await message.answer(
            f"file_id этого стикера:\n<code>{message.sticker.file_id}</code>\n\n"
            "Вставь его в переменную окружения PROMPT_STICKER_ID на Railway.",
            parse_mode="HTML",
        )


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
    user_id = message.from_user.id

    # Если у пользователя открыт мастер "Обложка трека" — обрабатываем текст по шагам
    cover_state = user_cover_state.get(user_id)
    if cover_state is not None:
        step = cover_state.get("step")
        if step == "lyrics":
            cover_state["lyrics"] = message.text
            cover_state["step"] = "photo_choice"
            await message.answer(COVER_PHOTO_CHOICE_TEXT, reply_markup=cover_photo_choice_keyboard())
        elif step == "awaiting_photo":
            await message.answer(
                "Пришли именно фото 📸 — или нажми кнопку, чтобы продолжить без него.",
                reply_markup=cover_awaiting_photo_keyboard(),
            )
        else:
            await message.answer("Выбери один из вариантов кнопками выше 👆")
        return

    # Если у пользователя включён режим перевода — обрабатываем текст иначе
    mode = user_translate_mode.get(user_id)
    if mode is not None:
        if is_rate_limited(user_id):
            await message.answer(
                f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
                f"запросов в {RATE_LIMIT_WINDOW} секунд)."
            )
            return

        # Пользователь вводит название/код языка после нажатия "Указать язык"
        if mode.get("awaiting_custom_lang"):
            custom_lang = message.text.strip()
            user_translate_mode[user_id] = {"target_lang": custom_lang}
            await message.answer(
                f"✅ Режим перевода на «{custom_lang}» включён.\n\nПросто присылай текст — переведу.",
                reply_markup=translate_active_keyboard(),
            )
            return

        # Обычный текст в активном режиме перевода — переводим и остаёмся в режиме
        translation = await with_typing(message.chat.id, translate_text(message.text, mode.get("target_lang")))
        await message.answer(f"🌐 {translation}", reply_markup=translate_active_keyboard())
        return

    # Если у пользователя включён режим составления промптов — ведём диалог по теме
    prompt_type = user_prompt_mode.get(user_id)
    if prompt_type is not None:
        if is_rate_limited(user_id):
            await message.answer(
                f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
                f"запросов в {RATE_LIMIT_WINDOW} секунд)."
            )
            return

        reply = await with_typing(message.chat.id, get_prompt_reply(user_id, prompt_type, message.text))
        await send_prompt_reply(message.chat.id, reply, prompt_active_keyboard())
        return

    if is_rate_limited(user_id):
        await message.answer(
            f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
            f"запросов в {RATE_LIMIT_WINDOW} секунд)."
        )
        return

    await bot.send_chat_action(message.chat.id, "typing")
    reply = await get_ai_reply(user_id, message.text)
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

    # Если пользователь на шаге "жду фото" в мастере "Обложка трека"
    cover_state = user_cover_state.get(message.from_user.id)
    if cover_state is not None and cover_state.get("step") == "awaiting_photo":
        cover_state["photo_base64"] = image_base64
        cover_state["step"] = "format"
        await message.answer(COVER_FORMAT_TEXT, reply_markup=cover_format_keyboard())
        return

    # Если у пользователя включён режим составления промпта для картинки —
    # обрабатываем фото как запрос на правку через генеративную нейросеть
    if user_prompt_mode.get(message.from_user.id) == "image":
        if not message.caption:
            await message.answer(
                "Пришли это фото ещё раз, но с подписью — опиши, что именно хочешь в нём поменять 🙂",
                reply_markup=prompt_active_keyboard(),
            )
            return

        reply = await with_typing(
            message.chat.id,
            get_image_prompt_from_photo(message.from_user.id, image_base64, message.caption),
        )
        await send_prompt_reply(message.chat.id, reply, prompt_active_keyboard())
        return

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
