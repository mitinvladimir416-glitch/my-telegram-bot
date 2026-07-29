import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
from datetime import date, timedelta
import httpx
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
from openai import OpenAI
from gtts import gTTS
from dotenv import load_dotenv

# Загружаем ключи из файла .env (он должен лежать в той же папке)
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # необязательный — DeepSeek через OpenRouter
ADMIN_ID = os.getenv("ADMIN_ID")  # твой Telegram ID — только ты сможешь делать рассылку
PROMPT_STICKER_ID = os.getenv("PROMPT_STICKER_ID")  # file_id стикера, который показывается перед готовым промптом

# Адрес botyara-api и секрет для связи с ним — чтобы сообщения и избранное из бота
# попадали в общую базу данных и были видны на сайте
BOTYARA_API_URL = os.getenv("BOTYARA_API_URL")
BOT_INTERNAL_SECRET = os.getenv("BOT_INTERNAL_SECRET")

if not TELEGRAM_TOKEN or not GROQ_API_KEY:
    raise ValueError(
        "Не найдены ключи! Проверь, что файл .env существует и содержит "
        "TELEGRAM_TOKEN и GROQ_API_KEY"
    )

ADMIN_ID = int(ADMIN_ID) if ADMIN_ID else None

MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.6-27b"
DEEPSEEK_MODEL = "deepseek/deepseek-v4-flash"  # основная модель для обычного чата, если ключ задан
HISTORY_FILE = "history.json"
USERS_FILE = "users.json"
FAVORITES_FILE = "favorites.json"
STATS_FILE = "stats.json"
ROLES_FILE = "user_roles.json"

# Лимит: не больше RATE_LIMIT_COUNT запросов за RATE_LIMIT_WINDOW секунд на пользователя
RATE_LIMIT_COUNT = 5
RATE_LIMIT_WINDOW = 60

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
# max_retries=0 — чтобы при лимите Groq бот сразу отвечал понятным сообщением,
# а не "молчал" по 20-30 секунд, пока SDK сам пытается повторить запрос
groq_client = Groq(api_key=GROQ_API_KEY, max_retries=0)

openrouter_client = (
    OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1", max_retries=0)
    if OPENROUTER_API_KEY
    else None
)


def chat_completion_with_fallback(messages: list, temperature: float = 0.85, max_tokens: int = 1024) -> str:
    """
    Пробует ответить через DeepSeek Flash (OpenRouter) — дешевле и часто качественнее для
    обычного текстового общения. Если ключ не задан или запрос не удался — прозрачно
    откатывается на Groq, чтобы бот не переставал работать.
    """
    if openrouter_client is not None:
        try:
            response = openrouter_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception:
            logging.exception("DeepSeek (OpenRouter) недоступен, переключаюсь на Groq")

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def _typing_loop(chat_id: int, action: str = "typing"):
    """Периодически шлёт индикатор действия (печатает.../записывает голосовое...), пока идёт
    долгий запрос к нейросети — иначе Telegram гасит индикатор через ~5 секунд."""
    try:
        while True:
            await bot.send_chat_action(chat_id, action)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


async def with_typing(chat_id: int, coro, action: str = "typing"):
    """Оборачивает долгий вызов нейросети, поддерживая индикатор действия всё время."""
    typing_task = asyncio.create_task(_typing_loop(chat_id, action))
    try:
        return await coro
    finally:
        typing_task.cancel()


async def send_prompt_reply(chat_id: int, reply_text: str, keyboard: InlineKeyboardMarkup):
    """Отправляет ответ из раздела 'Промпты'. Если это финальный готовый промпт —
    сначала на 3-4 секунды показывает стикер (для эффектности), потом удаляет его,
    и добавляет кнопку 'Сохранить в избранное'."""
    is_final = "ГОТОВЫЙ ПРОМПТ:" in reply_text

    if is_final:
        user_last_prompt[chat_id] = reply_text
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="⭐ Сохранить в избранное", callback_data="fav_save")]]
            + keyboard.inline_keyboard
        )

        if PROMPT_STICKER_ID:
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

# ==================== РОЛИ ДЛЯ ОБЩЕНИЯ (те же, что и на сайте) ====================

ROLE_COMMON_RULES = (
    "Общие правила общения (важно соблюдать всегда):\n"
    "- Говори живо, естественно, как реальный человек в переписке — короткими репликами, "
    "без канцелярита и занудных вступлений вроде 'Конечно! Вот что...'.\n"
    "- Реагируй эмоционально на то, что говорит собеседник — удивляйся, радуйся, сочувствуй, "
    "если это уместно по контексту.\n"
    "- Задавай встречные вопросы, поддерживай разговор, а не просто выдавай информацию.\n"
    "- Не упоминай, что ты нейросеть или языковая модель, если тебя прямо об этом не спросили.\n"
    "- Пиши без длинных списков и заголовков — это переписка, а не документ.\n"
    "- Используй уместный юмор и лёгкую иронию там, где это подходит по характеру роли.\n"
)

ROLE_CONFIG = {
    "default": {
        "label": "Обычное общение",
        "emoji": "🤖",
        "system_prompt": SYSTEM_PROMPT,
    },
    "friend": {
        "label": "Лучший друг",
        "emoji": "🧑‍🤝‍🧑",
        "system_prompt": (
            ROLE_COMMON_RULES
            + "\nТы — лучший друг пользователя. Общайся неформально, на \"ты\", с юмором и "
            "лёгким сленгом. Ты всегда на стороне собеседника, поддерживаешь его, но можешь и "
            "по-дружески подколоть. Искренне интересуешься, как у него дела."
        ),
    },
    "mentor": {
        "label": "Мудрый наставник",
        "emoji": "🧙",
        "system_prompt": (
            ROLE_COMMON_RULES
            + "\nТы — мудрый наставник с большим жизненным опытом. Говоришь спокойно и вдумчиво, "
            "иногда приводишь метафоры или короткие истории для примера. Не поучаешь свысока, "
            "а делишься опытом на равных. Помогаешь увидеть ситуацию под другим углом."
        ),
    },
    "listener": {
        "label": "Внимательный собеседник",
        "emoji": "🕊",
        "system_prompt": (
            ROLE_COMMON_RULES
            + "\nТы — тёплый, эмпатичный собеседник, который умеет слушать. Не оцениваешь и не "
            "критикуешь, отражаешь чувства собеседника, задаёшь мягкие уточняющие вопросы. "
            "Создаёшь ощущение, что его действительно слышат."
        ),
    },
    "wit": {
        "label": "Остроумный циник",
        "emoji": "😏",
        "system_prompt": (
            ROLE_COMMON_RULES
            + "\nТы — остроумный собеседник с сухим, слегка циничным чувством юмора. Любишь "
            "подколоть, пошутить с сарказмом, но не переходишь на откровенную грубость или "
            "оскорбления. За иронией видна теплота к собеседнику."
        ),
    },
    "motivator": {
        "label": "Мотиватор",
        "emoji": "🔥",
        "system_prompt": (
            ROLE_COMMON_RULES
            + "\nТы — заряженный энергией мотиватор. Веришь в собеседника больше, чем он сам "
            "в себя, подбадриваешь, помогаешь увидеть возможности, а не препятствия. Говоришь "
            "энергично, но не переигрывай в наигранный пафос — искренне и по делу."
        ),
    },
    "flirty": {
        "label": "Лёгкий флирт",
        "emoji": "😉",
        "system_prompt": (
            ROLE_COMMON_RULES
            + "\nТы — игривый, обаятельный собеседник, который любит лёгкий флирт: комплименты, "
            "дружеские подколки, немного интриги в тоне. Держись в рамках приличия — флирт лёгкий "
            "и уважительный, без пошлости и явного сексуального содержания."
        ),
    },
}

DEFAULT_ROLE = "default"

CATEGORY_LABELS = {
    "suno": "🎵 Suno",
    "image": "🖼 Картинка",
    "video": "🎬 Видео",
    "cover": "🖼️ Обложка трека",
    "other": "💬 Разное",
}

# ==================== ОПОВЕЩЕНИЯ ОБ ОБНОВЛЕНИЯХ (для админа) ====================

ANNOUNCE_SYSTEM_PROMPT = (
    "Ты — автор дружелюбных, живых постов об обновлениях Telegram-бота «Ботяра» "
    "(нейро-помощник: общение, переводчик, промпты для музыки/картинок/видео, обложки треков). "
    "Стиль бота — неформальный, с юмором, на \"ты\", уместные эмодзи, без канцелярита и воды.\n\n"
    "Тебе дают список того, что изменилось (иногда сухими фразами или просто списком) — "
    "превращай это в один цельный, воодушевляющий пост для рассылки пользователям бота.\n\n"
    "Правила:\n"
    "- НЕ используй HTML-теги (<b>, <i> и т.п.) — они не поддерживаются, пиши только простым текстом.\n"
    "- Используй эмодзи по смыслу, но не перегружай ими.\n"
    "- Короткие абзацы, разделяй их пустой строкой.\n"
    "- Заверши бодрым призывом попробовать новое прямо сейчас.\n"
    "- Ответь только готовым текстом поста, без пояснений от себя."
)


async def generate_announcement(raw_notes: str) -> str:
    """Просит нейросеть красиво оформить список изменений в пост для рассылки пользователям."""
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ANNOUNCE_SYSTEM_PROMPT},
                {"role": "user", "content": raw_notes},
            ],
            temperature=0.8,
            max_tokens=800,
        )
        return clean_reply(response.choices[0].message.content)
    except Exception as e:
        logging.exception("Ошибка при генерации оповещения об обновлении")
        return describe_groq_error(e)


def announce_confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Разослать всем", callback_data="announce_send")],
            [InlineKeyboardButton(text="🔄 Переписать заново", callback_data="announce_retry")],
            [InlineKeyboardButton(text="❌ Отмена", callback_data="announce_cancel")],
        ]
    )

WELCOME_IMAGE = "welcome.jpg"
WELCOME_TEXT = (
    "🤖 <b>ЙО! ДОБРО ПОЖАЛОВАТЬ В МИР БОТЯРЫ!</b>\n\n"
    "Нажал старт — красавчик, теперь ты в деле 😈\n\n"
    "Что я умею:\n"
    "💬 Шарю за любые темы — просто пиши\n"
    "🎭 Можешь выбрать роль общения (друг, наставник, мотиватор и другие) — в разделе «Общение»\n"
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
    """
    Загружает историю диалогов из файла при запуске бота.
    Формат: user_id -> {role_id: [сообщения]} — отдельная история на каждую роль,
    как вкладки на сайте. Если в файле старый формат (просто список сообщений на
    пользователя, без ролей) — переносим его в роль "default", чтобы не потерять историю.
    """
    raw = load_json_file(HISTORY_FILE, {})
    result = {}
    for k, v in raw.items():
        user_id = int(k)
        if isinstance(v, list):
            result[user_id] = {"default": v}
        else:
            result[user_id] = v
    return result


def save_histories():
    """Сохраняет текущую историю диалогов (по всем ролям) в файл."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(user_histories, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.exception(f"НЕ УДАЛОСЬ сохранить историю: {e}")


def get_user_history(user_id: int, role: str) -> list:
    """Возвращает (создавая при необходимости) список сообщений конкретного пользователя
    для конкретной роли — отдельная история на каждую роль, не смешиваются."""
    user_data = user_histories.setdefault(user_id, {})
    return user_data.setdefault(role, [])


def load_user_roles() -> dict:
    """Загружает, какую роль общения выбрал каждый пользователь в последний раз."""
    raw = load_json_file(ROLES_FILE, {})
    return {int(k): v for k, v in raw.items()}


def save_user_roles():
    """Сохраняет выбранные роли пользователей в файл."""
    try:
        with open(ROLES_FILE, "w", encoding="utf-8") as f:
            json.dump(user_roles, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.exception(f"НЕ УДАЛОСЬ сохранить роли пользователей: {e}")


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


def load_favorites() -> dict:
    """Загружает сохранённые промпты пользователей из файла."""
    raw = load_json_file(FAVORITES_FILE, {})
    return {int(k): v for k, v in raw.items()}


def save_favorites():
    """Сохраняет избранные промпты в файл."""
    try:
        with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
            json.dump(user_favorites, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.exception(f"НЕ УДАЛОСЬ сохранить избранное: {e}")


def load_stats() -> dict:
    """Загружает статистику использования бота из файла."""
    default = {"messages_total": 0, "messages_by_date": {}, "sections": {}}
    raw = load_json_file(STATS_FILE, default)
    # На случай если в старом файле не хватает какого-то ключа
    for key, value in default.items():
        raw.setdefault(key, value)
    return raw


def save_stats():
    """Сохраняет статистику использования бота в файл."""
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.exception(f"НЕ УДАЛОСЬ сохранить статистику: {e}")


async def sync_message_to_site(
    user_id: int, username: str | None, first_name: str | None, role: str, content: str, persona: str = "default"
):
    """
    Отправляет одно сообщение в botyara-api, чтобы оно попало в общую историю и было
    видно на сайте. Если сайт временно недоступен — просто пишем в лог и не мешаем боту
    работать дальше (история всё равно останется в локальном history.json).
    persona — какая роль/вкладка общения (см. ROLE_CONFIG), не путать с role ("user"/"assistant").
    """
    if not BOTYARA_API_URL or not BOT_INTERNAL_SECRET:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{BOTYARA_API_URL}/api/bot/message",
                json={
                    "telegram_id": user_id,
                    "telegram_username": username,
                    "telegram_first_name": first_name,
                    "role": role,
                    "content": content,
                    "persona": persona,
                },
                headers={"X-Bot-Secret": BOT_INTERNAL_SECRET},
            )
    except Exception:
        logging.exception("Не удалось отправить сообщение на сайт (не критично, бот продолжает работать)")


async def sync_favorite_to_site(
    user_id: int, username: str | None, first_name: str | None, content: str, category: str = "other"
):
    """Отправляет сохранённый промпт в общее избранное на сайте."""
    if not BOTYARA_API_URL or not BOT_INTERNAL_SECRET:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{BOTYARA_API_URL}/api/bot/favorite",
                json={
                    "telegram_id": user_id,
                    "telegram_username": username,
                    "telegram_first_name": first_name,
                    "content": content,
                    "category": category,
                },
                headers={"X-Bot-Secret": BOT_INTERNAL_SECRET},
            )
    except Exception:
        logging.exception("Не удалось отправить избранное на сайт (не критично, бот продолжает работать)")


async def fetch_gallery_list(limit: int = 15) -> list[dict] | None:
    """Список последних опубликованных промптов галереи. None — если сайт недоступен."""
    if not BOTYARA_API_URL or not BOT_INTERNAL_SECRET:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{BOTYARA_API_URL}/api/bot/gallery",
                params={"limit": limit},
                headers={"X-Bot-Secret": BOT_INTERNAL_SECRET},
            )
            resp.raise_for_status()
            return resp.json().get("posts", [])
    except Exception:
        logging.exception("Не удалось получить список галереи")
        return None


async def fetch_gallery_post(post_id: int) -> dict | None:
    """Один пост галереи с комментариями. None — если сайт недоступен или пост не найден."""
    if not BOTYARA_API_URL or not BOT_INTERNAL_SECRET:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{BOTYARA_API_URL}/api/bot/gallery/{post_id}",
                headers={"X-Bot-Secret": BOT_INTERNAL_SECRET},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception:
        logging.exception("Не удалось получить пост галереи")
        return None


async def publish_to_gallery(
    user_id: int, username: str | None, first_name: str | None, content: str, category: str = "other"
) -> dict | None:
    """Публикует промпт в галерею (проходит модерацию на бэкенде). None — если сайт недоступен."""
    if not BOTYARA_API_URL or not BOT_INTERNAL_SECRET:
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{BOTYARA_API_URL}/api/bot/gallery/publish",
                json={
                    "telegram_id": user_id,
                    "telegram_username": username,
                    "telegram_first_name": first_name,
                    "content": content,
                    "category": category,
                },
                headers={"X-Bot-Secret": BOT_INTERNAL_SECRET},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception:
        logging.exception("Не удалось опубликовать в галерею")
        return None


async def comment_on_gallery(
    user_id: int, username: str | None, first_name: str | None, post_id: int, content: str
) -> dict | None:
    """Отправляет комментарий к посту галереи (тоже проходит модерацию). None — если сайт недоступен."""
    if not BOTYARA_API_URL or not BOT_INTERNAL_SECRET:
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{BOTYARA_API_URL}/api/bot/gallery/comment",
                json={
                    "telegram_id": user_id,
                    "telegram_username": username,
                    "telegram_first_name": first_name,
                    "post_id": post_id,
                    "content": content,
                },
                headers={"X-Bot-Secret": BOT_INTERNAL_SECRET},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception:
        logging.exception("Не удалось отправить комментарий в галерею")
        return None


async def fetch_public_chat(limit: int = 15) -> list[dict] | None:
    """Последние сообщения общего публичного чата. None — если сайт недоступен."""
    if not BOTYARA_API_URL or not BOT_INTERNAL_SECRET:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{BOTYARA_API_URL}/api/bot/public-chat",
                params={"limit": limit},
                headers={"X-Bot-Secret": BOT_INTERNAL_SECRET},
            )
            resp.raise_for_status()
            return resp.json().get("messages", [])
    except Exception:
        logging.exception("Не удалось получить сообщения общего чата")
        return None


async def send_public_chat(user_id: int, username: str | None, first_name: str | None, content: str) -> dict | None:
    """Отправляет сообщение в общий публичный чат (проходит модерацию). None — если сайт недоступен."""
    if not BOTYARA_API_URL or not BOT_INTERNAL_SECRET:
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{BOTYARA_API_URL}/api/bot/public-chat",
                json={
                    "telegram_id": user_id,
                    "telegram_username": username,
                    "telegram_first_name": first_name,
                    "content": content,
                },
                headers={"X-Bot-Secret": BOT_INTERNAL_SECRET},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception:
        logging.exception("Не удалось отправить сообщение в общий чат")
        return None


async def save_announcement_to_site(content: str) -> None:
    """Сохраняет оповещение об обновлении на сайте (лента в разделе уведомлений)."""
    if not BOTYARA_API_URL or not BOT_INTERNAL_SECRET:
        return
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(
                f"{BOTYARA_API_URL}/api/bot/announcements",
                json={"content": content},
                headers={"X-Bot-Secret": BOT_INTERNAL_SECRET},
            )
    except Exception:
        logging.exception("Не удалось сохранить оповещение на сайте (не критично)")


def record_message_stat():
    """Учитывает одно входящее сообщение (любого типа) в статистике."""
    stats["messages_total"] += 1
    today = date.today().isoformat()
    stats["messages_by_date"][today] = stats["messages_by_date"].get(today, 0) + 1
    save_stats()


def record_section_stat(section: str):
    """Учитывает открытие раздела меню в статистике."""
    stats["sections"][section] = stats["sections"].get(section, 0) + 1
    save_stats()


# Память диалога и список пользователей — подгружаются из файлов при старте
user_histories = load_histories()
registered_users = load_users()
user_favorites = load_favorites()  # user_id -> список сохранённых промптов
user_roles = load_user_roles()  # user_id -> id выбранной роли общения (см. ROLE_CONFIG)
stats = load_stats()

# Храним последний сгенерированный промпт каждого пользователя — чтобы кнопка
# "Сохранить в избранное" знала, что именно сохранять (в файл не пишем, это временное)
user_last_prompt: dict[int, str] = {}

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

# Присланные кадры для темы "video": user_id -> {"first": base64|None, "last": base64|None, "note": str}
user_video_frames: dict[int, dict] = {}
user_prompt_histories: dict[int, list] = {}
user_prompt_target: dict[int, str] = {}  # user_id -> выбранная версия/нейросеть (текст для показа)

# Храним состояние мастера "Обложка трека": шаг, текст песни, фото (если есть)
user_cover_state: dict[int, dict] = {}

# Храним, у кого включена озвучка ответов в обычном общении (по умолчанию выключена)
user_tts_enabled: dict[int, bool] = {}

# Черновик оповещения об обновлении, которое администратор готовит перед рассылкой:
# admin_id -> {"raw": исходный текст, "text": текст, оформленный нейросетью}
user_pending_announcement: dict[int, dict] = {}

# Категория (папка) для каждого сохранённого в избранное промпта: user_id -> {текст: категория}
# Храним отдельно от user_favorites, чтобы не переделывать формат history/favorites.json
user_favorite_categories: dict[int, dict[str, str]] = {}

# Пользователь нажал "Написать комментарий" к посту галереи — ждём от него текстовое сообщение:
# user_id -> id поста, к которому пишется комментарий
user_gallery_comment_target: dict[int, int] = {}

# Пользователь нажал "Написать в общий чат" — ждём от него текстовое сообщение
user_publicchat_pending: set[int] = set()


def synthesize_speech_sync(text: str) -> str:
    """Синхронно генерирует mp3-файл с озвучкой текста (для запуска в отдельном потоке)."""
    # gTTS не любит слишком длинный текст и падает на некоторых спецсимволах — подчищаем
    clean_text = re.sub(r"[*_`#]", "", text).strip()
    if len(clean_text) > 3000:
        clean_text = clean_text[:3000]

    tts = gTTS(text=clean_text, lang="ru")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(tmp.name)
    return tmp.name


async def send_voice_reply(chat_id: int, text: str):
    """Озвучивает текст и отправляет как голосовое/аудио сообщение."""
    try:
        mp3_path = await with_typing(
            chat_id, asyncio.to_thread(synthesize_speech_sync, text), action="record_voice"
        )
    except Exception:
        logging.exception("Не удалось сгенерировать озвучку ответа")
        return

    try:
        audio_file = FSInputFile(mp3_path)
        await bot.send_audio(chat_id, audio_file, title="Ответ Ботяры 🎙")
    except Exception:
        logging.exception("Не удалось отправить озвученный ответ")
    finally:
        try:
            os.remove(mp3_path)
        except OSError:
            pass


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="💬 Общение", callback_data="menu_chat")],
            [InlineKeyboardButton(text="🌐 Переводчик", callback_data="menu_translate")],
            [InlineKeyboardButton(text="🎨 Промпты", callback_data="menu_prompts")],
            [InlineKeyboardButton(text="🖼 Обложка трека", callback_data="menu_cover")],
            [InlineKeyboardButton(text="⭐ Избранное", callback_data="menu_favorites")],
            [InlineKeyboardButton(text="🖼️ Галерея", callback_data="menu_gallery")],
            [InlineKeyboardButton(text="🌍 Общий чат", callback_data="menu_publicchat")],
            [InlineKeyboardButton(text="ℹ️ Помощь", callback_data="menu_help")],
        ]
    )


def main_menu_button_keyboard() -> InlineKeyboardMarkup:
    """Маленькая клавиатура из одной кнопки — вернуться в главное меню.
    Используется везде, где бот отвечает вне контекста разделов (обычный чат, команды)."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="🏠 Главное меню", callback_data="menu_back")]]
    )


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")]]
    )


def chat_settings_keyboard(user_id: int) -> InlineKeyboardMarkup:
    voice_on = user_tts_enabled.get(user_id, False)
    text_label = "✅ 💬 Текстом" if not voice_on else "⚪ 💬 Текстом"
    voice_label = "✅ 🎙 Голосом" if voice_on else "⚪ 🎙 Голосом"
    role = user_roles.get(user_id, DEFAULT_ROLE)
    role_cfg = ROLE_CONFIG.get(role, ROLE_CONFIG[DEFAULT_ROLE])
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=text_label, callback_data="chat_mode_text"),
                InlineKeyboardButton(text=voice_label, callback_data="chat_mode_voice"),
            ],
            [
                InlineKeyboardButton(
                    text=f"🎭 Роль: {role_cfg['emoji']} {role_cfg['label']}", callback_data="chat_role_menu"
                )
            ],
            [InlineKeyboardButton(text="🗑 Начать эту роль заново", callback_data="chat_role_reset")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
        ]
    )


def role_menu_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Клавиатура выбора роли общения — по две в ряд, текущая отмечена галочкой."""
    current = user_roles.get(user_id, DEFAULT_ROLE)
    items = list(ROLE_CONFIG.items())
    rows = []
    for i in range(0, len(items), 2):
        row = []
        for role_id, cfg in items[i : i + 2]:
            mark = "✅ " if role_id == current else ""
            row.append(
                InlineKeyboardButton(text=f"{mark}{cfg['emoji']} {cfg['label']}", callback_data=f"set_role_{role_id}")
            )
        rows.append(row)
    rows.append([InlineKeyboardButton(text="◀️ Назад", callback_data="menu_chat")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


MENU_MAIN_TEXT = "📋 <b>Главное меню</b>\n\nВыбери, что хочешь сделать:"


def chat_menu_text(user_id: int) -> str:
    role = user_roles.get(user_id, DEFAULT_ROLE)
    role_cfg = ROLE_CONFIG.get(role, ROLE_CONFIG[DEFAULT_ROLE])
    return (
        "💬 <b>Режим общения</b>\n\n"
        f"Сейчас общаешься с ролью: {role_cfg['emoji']} <b>{role_cfg['label']}</b>\n"
        "Пиши мне текстом, голосом или фото — отвечу с помощью нейросети.\n\n"
        "💡 Кстати, то же самое общение (и вся история) доступно и на сайте 24promtbot.ru — "
        "там ещё удобнее с телефона или компьютера.\n\n"
        "Настрой ниже 👇"
    )


ROLE_MENU_TEXT = (
    "🎭 <b>С кем хочешь общаться?</b>\n\n"
    "У каждой роли своя манера речи и своя отдельная история — как вкладки на сайте, они не смешиваются."
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

# Храним выбранный тип перевода: "text" (по умолчанию) или "voice"
user_translate_input_type: dict[int, str] = {}

TRANSLATE_SUBMENU_TEXT = (
    "🌐 <b>Переводчик</b>\n\n"
    "Сначала выбери, что переводим — текст или голосовые сообщения, а затем язык:"
)


def translate_submenu_keyboard(user_id: int) -> InlineKeyboardMarkup:
    input_type = user_translate_input_type.get(user_id, "text")
    text_label = "✅ ⌨️ Текста" if input_type == "text" else "⚪ ⌨️ Текста"
    voice_label = "✅ 🎙 Голосовых" if input_type == "voice" else "⚪ 🎙 Голосовых"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=text_label, callback_data="tr_input_text"),
                InlineKeyboardButton(text=voice_label, callback_data="tr_input_voice"),
            ],
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
            "✅ Нейросеть: <b>{target}</b>\n\nОпиши идею сцены.\n\n"
            "💡 Можно и по-другому: пришли картинкой первый (и, если есть, последний) кадр сцены — "
            "учту их визуально при составлении промпта. Пришлёшь только один кадр — тоже сработает, "
            "просто опиши остальное словами."
        ),
        "system_prompt": (
            "Ты — эксперт по составлению промптов для генерации видео через нейросети "
            "(Sora, Runway, Kling, Veo, Pika и подобные). Твоя задача — помочь пользователю "
            "составить качественный промпт. Веди диалог по существу:\n"
            "0. Если пользователь ещё не присылал изображения кадров, в самом начале обязательно "
            "спроси: есть ли у него референсные кадры — первый и/или последний кадр сцены "
            "(можно прислать картинками, это очень поможет с деталями). Если кадров нет — "
            "предложи просто описать сцену словами и продолжай без них.\n"
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
        + [[InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")]]
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

    rows.append([InlineKeyboardButton(text="◀️ К темам", callback_data="menu_prompts")])
    rows.append([InlineKeyboardButton(text="🏠 Главное меню", callback_data="menu_back")])

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


async def get_video_prompt_from_frames(
    user_id: int, description: str, first_b64: str | None, last_b64: str | None
) -> str:
    """Составляет видео-промпт по присланным кадрам (первому и/или последнему) и описанию словами."""
    config = PROMPT_CONFIG["video"]
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
        + "\n\nПользователь прислал референсные кадры сцены (первый и/или последний кадр видео). "
        + target_note
        + "Учти визуальные детали кадров при составлении промпта, сразу выдай готовый промпт "
        "с пометкой 'ГОТОВЫЙ ПРОМПТ:'."
    )

    content = [{"type": "text", "text": description or "Опиши сцену по присланным кадрам."}]
    if first_b64:
        content.append({"type": "text", "text": "Это первый кадр сцены:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{first_b64}"}})
    if last_b64:
        content.append({"type": "text", "text": "Это последний кадр сцены:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{last_b64}"}})

    trimmed_history = history[-10:]
    messages = [{"role": "system", "content": combined_system}] + trimmed_history + [
        {"role": "user", "content": content}
    ]

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
        logging.exception("Ошибка при составлении промпта по кадрам видео")
        reply = describe_groq_error(e)

    history.append({"role": "user", "content": f"[Прислал кадры видео] {description}"})
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

COVER_TEXT_CHOICE_TEXT = (
    "✅ Принято.\n\n"
    "Нужен ли текст на самой обложке (например, название трека и/или имя исполнителя)?"
)

COVER_AWAITING_TEXT_TEXT = "✏️ Напиши текст, который должен быть на обложке (например: название трека — исполнитель)."

COVER_FORMAT_TEXT = "📐 Теперь выбери формат обложки:"

COVER_PROMPT_FOOTER = "\n\n💡 Промпт составлен для ChatGPT Image 2"

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
    "(лирику), возможно — референсное фото, возможно — текст для размещения на обложке (название "
    "трека/исполнителя), и нужное соотношение сторон обложки. Твоя задача:\n"
    "1. Проанализируй текст песни — определи настроение, тематику, ключевые образы и символы.\n"
    "2. Если есть референсное фото — учти его стиль, цветовую палитру, композицию.\n"
    "3. Составь единый выразительный промпт для обложки: визуальная композиция, художественный стиль "
    "(фотореализм/иллюстрация/абстракция и т.д.), цветовая палитра, настроение, ключевые визуальные "
    "элементы.\n"
    "4. Если пользователь указал текст для обложки — обязательно включи в промпт точную инструкцию "
    "разместить именно этот текст (шрифт, расположение). Если текста нет — обложка должна быть БЕЗ "
    "текста и букв.\n"
    "Сначала коротко (1-2 предложения на русском) опиши, какое настроение считал из лирики. "
    "Затем сразу выдай готовый промпт под пометкой 'ГОТОВЫЙ ПРОМПТ:' на отдельной строке — "
    "сам промпт пиши на английском (так эффективнее для генерации), обязательно укажи в конце "
    "нужный aspect ratio."
)


def cover_back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")]]
    )


def cover_photo_choice_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📸 Да, пришлю фото", callback_data="cover_photo_yes")],
            [InlineKeyboardButton(text="⏭ Без фото", callback_data="cover_photo_no")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
        ]
    )


def cover_awaiting_photo_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⏭ Продолжить без фото", callback_data="cover_photo_no")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
        ]
    )


def cover_text_choice_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✏️ Указать текст", callback_data="cover_text_yes")],
            [InlineKeyboardButton(text="🚫 Без текста", callback_data="cover_text_no")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
        ]
    )


def cover_awaiting_text_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🚫 Передумал, без текста", callback_data="cover_text_no")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
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
    rows.append([InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def cover_result_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Новая обложка", callback_data="menu_cover")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
        ]
    )


async def generate_cover_prompt(
    lyrics: str, photo_base64: str | None, ratio: str, cover_text: str | None
) -> str:
    """Анализирует текст песни (и фото, если есть) и составляет промпт для обложки трека."""
    user_content_text = f"Текст песни:\n{lyrics}\n\nНужное соотношение сторон: {ratio}"
    if cover_text:
        user_content_text += f"\n\nТекст, который должен быть на обложке: {cover_text}"
    else:
        user_content_text += "\n\nТекста на обложке быть не должно."

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
        reply = clean_reply(response.choices[0].message.content)
    except Exception as e:
        logging.exception("Ошибка при составлении промпта обложки")
        return describe_groq_error(e)

    return reply + COVER_PROMPT_FOOTER



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
    record_section_stat("chat")
    user_translate_mode.pop(callback.from_user.id, None)
    user_prompt_mode.pop(callback.from_user.id, None)
    user_prompt_target.pop(callback.from_user.id, None)
    user_cover_state.pop(callback.from_user.id, None)
    await show_menu_screen(callback, chat_menu_text(callback.from_user.id), chat_settings_keyboard(callback.from_user.id))


@dp.callback_query(F.data.in_(["chat_mode_text", "chat_mode_voice"]))
async def callback_chat_mode(callback: CallbackQuery):
    user_id = callback.from_user.id
    user_tts_enabled[user_id] = callback.data == "chat_mode_voice"
    await show_menu_screen(callback, chat_menu_text(user_id), chat_settings_keyboard(user_id))


@dp.callback_query(F.data == "chat_role_menu")
async def callback_chat_role_menu(callback: CallbackQuery):
    await show_menu_screen(callback, ROLE_MENU_TEXT, role_menu_keyboard(callback.from_user.id))


@dp.callback_query(F.data.startswith("set_role_"))
async def callback_set_role(callback: CallbackQuery):
    role_id = callback.data[len("set_role_"):]
    if role_id not in ROLE_CONFIG:
        await callback.answer("Неизвестная роль", show_alert=True)
        return
    user_id = callback.from_user.id
    user_roles[user_id] = role_id
    save_user_roles()
    await callback.answer(f"Роль изменена: {ROLE_CONFIG[role_id]['label']}")
    await show_menu_screen(callback, chat_menu_text(user_id), chat_settings_keyboard(user_id))


@dp.callback_query(F.data == "chat_role_reset")
async def callback_chat_role_reset(callback: CallbackQuery):
    user_id = callback.from_user.id
    role = user_roles.get(user_id, DEFAULT_ROLE)
    get_user_history(user_id, role).clear()
    save_histories()
    await callback.answer("История этой роли очищена ✅")


@dp.callback_query(F.data == "menu_translate")
async def callback_menu_translate(callback: CallbackQuery):
    record_section_stat("translate")
    # Выходим из режима перевода/промптов при повторном открытии подменю
    user_translate_mode.pop(callback.from_user.id, None)
    user_prompt_mode.pop(callback.from_user.id, None)
    user_prompt_target.pop(callback.from_user.id, None)
    user_cover_state.pop(callback.from_user.id, None)
    await show_menu_screen(callback, TRANSLATE_SUBMENU_TEXT, translate_submenu_keyboard(callback.from_user.id))


@dp.callback_query(F.data.in_(["tr_input_text", "tr_input_voice"]))
async def callback_translate_input_type(callback: CallbackQuery):
    user_id = callback.from_user.id
    user_translate_input_type[user_id] = "voice" if callback.data == "tr_input_voice" else "text"
    await show_menu_screen(callback, TRANSLATE_SUBMENU_TEXT, translate_submenu_keyboard(user_id))


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


@dp.callback_query(F.data == "fav_save")
async def callback_fav_save(callback: CallbackQuery):
    user_id = callback.from_user.id
    prompt_text = user_last_prompt.get(user_id)

    if not prompt_text:
        await callback.answer("Нечего сохранять — сначала сгенерируй промпт.", show_alert=True)
        return

    favorites = user_favorites.setdefault(user_id, [])
    if prompt_text in favorites:
        await callback.answer("Этот промпт уже в избранном ⭐", show_alert=True)
        return

    favorites.append(prompt_text)
    # Ограничиваем на всякий случай, чтобы список не разрастался бесконечно
    user_favorites[user_id] = favorites[-30:]
    save_favorites()

    category = user_prompt_mode.get(user_id, "other")
    user_favorite_categories.setdefault(user_id, {})[prompt_text] = category

    await sync_favorite_to_site(
        user_id,
        callback.from_user.username,
        callback.from_user.first_name,
        prompt_text,
        category=category,
    )

    await callback.answer("Сохранено в избранное ⭐")


def favorites_list_keyboard(favorites: list[str]) -> InlineKeyboardMarkup:
    rows = []
    for i in range(len(favorites)):
        rows.append(
            [InlineKeyboardButton(text=f"📢 Опубликовать #{i + 1} в галерею", callback_data=f"gallery_pub_{i}")]
        )
    if favorites:
        rows.append([InlineKeyboardButton(text="🗑 Очистить всё", callback_data="fav_clear")])
    rows.append([InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@dp.callback_query(F.data == "menu_favorites")
async def callback_menu_favorites(callback: CallbackQuery):
    record_section_stat("favorites")
    user_id = callback.from_user.id
    favorites = user_favorites.get(user_id, [])

    if not favorites:
        text = "⭐ <b>Избранное</b>\n\nПока пусто. Сохраняй промпты кнопкой ⭐ под готовым результатом."
        await show_menu_screen(callback, text, favorites_list_keyboard([]))
        return

    # Показываем список коротким превью — полный текст лучше смотреть в самих сохранённых сообщениях
    lines = ["⭐ <b>Избранное</b>\n"]
    cats = user_favorite_categories.get(user_id, {})
    for i, item in enumerate(favorites, start=1):
        preview = item.replace("\n", " ").strip()
        if len(preview) > 80:
            preview = preview[:80] + "…"
        cat_label = CATEGORY_LABELS.get(cats.get(item, "other"), CATEGORY_LABELS["other"])
        lines.append(f"{i}. {cat_label} · {preview}")
    text = "\n".join(lines)

    await show_menu_screen(callback, text, favorites_list_keyboard(favorites))


@dp.callback_query(F.data == "fav_clear")
async def callback_fav_clear(callback: CallbackQuery):
    user_id = callback.from_user.id
    user_favorites[user_id] = []
    save_favorites()
    text = "⭐ <b>Избранное</b>\n\nПока пусто. Сохраняй промпты кнопкой ⭐ под готовым результатом."
    await show_menu_screen(callback, text, favorites_list_keyboard([]))


@dp.callback_query(F.data.startswith("gallery_pub_"))
async def callback_gallery_publish_from_favorites(callback: CallbackQuery):
    """Публикует конкретный промпт из избранного в общую галерею (с модерацией на бэкенде)."""
    user_id = callback.from_user.id
    idx = int(callback.data.rsplit("_", 1)[-1])
    favorites = user_favorites.get(user_id, [])
    if idx < 0 or idx >= len(favorites):
        await callback.answer("Промпт не найден", show_alert=True)
        return

    await callback.answer("Публикую…")
    category = user_favorite_categories.get(user_id, {}).get(favorites[idx], "other")
    result = await publish_to_gallery(
        user_id, callback.from_user.username, callback.from_user.first_name, favorites[idx], category=category
    )
    if result is None:
        await callback.message.answer("Галерея временно недоступна — попробуй чуть позже.")
    elif result.get("status") == "approved":
        await callback.message.answer("✅ Опубликовано в галерее! Загляни в раздел «🖼️ Галерея».")
    else:
        await callback.message.answer(
            f"🚫 Отклонено модерацией: {result.get('reject_reason') or 'нарушение правил платформы'}"
        )


# ==================== Галерея промптов ====================

def gallery_list_keyboard(posts: list[dict]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=f"Открыть #{p['id']} · 💬{p['comment_count']}", callback_data=f"gallery_open_{p['id']}")]
        for p in posts
    ]
    rows.append([InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def gallery_post_keyboard(post_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="💬 Написать комментарий", callback_data=f"gallery_comment_{post_id}")],
            [InlineKeyboardButton(text="◀️ К галерее", callback_data="menu_gallery")],
        ]
    )


@dp.callback_query(F.data == "menu_gallery")
async def callback_menu_gallery(callback: CallbackQuery):
    record_section_stat("gallery")
    posts = await fetch_gallery_list(15)
    if posts is None:
        await callback.answer("Галерея временно недоступна", show_alert=True)
        return
    if not posts:
        text = "🖼️ <b>Галерея</b>\n\nПока пусто — опубликуй что-нибудь из своего Избранного!"
        await show_menu_screen(callback, text, back_keyboard())
        return

    lines = ["🖼️ <b>Галерея</b>\n"]
    for p in posts:
        preview = p["content"].replace("\n", " ").strip()
        if len(preview) > 70:
            preview = preview[:70] + "…"
        cat_label = CATEGORY_LABELS.get(p.get("category", "other"), CATEGORY_LABELS["other"])
        lines.append(f"#{p['id']} · {cat_label} · {p['author']}: {preview}")
    text = "\n".join(lines)

    await show_menu_screen(callback, text, gallery_list_keyboard(posts))


@dp.callback_query(F.data.startswith("gallery_open_"))
async def callback_gallery_open(callback: CallbackQuery):
    post_id = int(callback.data.rsplit("_", 1)[-1])
    post = await fetch_gallery_post(post_id)
    if post is None:
        await callback.answer("Не удалось загрузить пост", show_alert=True)
        return

    lines = [f"🖼️ Пост от {post['author']}:\n", post["content"], f"\n💬 Комментарии ({len(post['comments'])}):"]
    for c in post["comments"][-10:]:
        lines.append(f"— {c['author']}: {c['content']}")
    text = "\n".join(lines)

    await show_menu_screen(callback, text, gallery_post_keyboard(post_id))


@dp.callback_query(F.data.startswith("gallery_comment_"))
async def callback_gallery_comment_start(callback: CallbackQuery):
    post_id = int(callback.data.rsplit("_", 1)[-1])
    user_gallery_comment_target[callback.from_user.id] = post_id
    await callback.answer()
    await callback.message.answer("✏️ Напиши текст комментария следующим сообщением.")


# ==================== Общий публичный чат ====================

def public_chat_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✍️ Написать сообщение", callback_data="publicchat_write")],
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="menu_publicchat")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_back")],
        ]
    )


@dp.callback_query(F.data == "menu_publicchat")
async def callback_menu_publicchat(callback: CallbackQuery):
    record_section_stat("publicchat")
    messages = await fetch_public_chat(15)
    if messages is None:
        await callback.answer("Общий чат временно недоступен", show_alert=True)
        return

    if not messages:
        text = "🌍 <b>Общий чат</b>\n\nПока пусто — все сообщения здесь видны всем пользователям бота."
    else:
        lines = ["🌍 <b>Общий чат</b>\n"]
        for m in messages:
            lines.append(f"<b>{m['author']}:</b> {m['content']}")
        text = "\n".join(lines)

    await show_menu_screen(callback, text, public_chat_keyboard())


@dp.callback_query(F.data == "publicchat_write")
async def callback_publicchat_write(callback: CallbackQuery):
    user_publicchat_pending.add(callback.from_user.id)
    await callback.answer()
    await callback.message.answer(
        "✏️ Напиши сообщение следующим текстом — его увидят все пользователи в общем чате."
    )


@dp.callback_query(F.data == "menu_prompts")
async def callback_menu_prompts(callback: CallbackQuery):
    record_section_stat("prompts")
    user_prompt_mode.pop(callback.from_user.id, None)
    user_prompt_target.pop(callback.from_user.id, None)
    user_cover_state.pop(callback.from_user.id, None)
    await show_menu_screen(callback, PROMPTS_SUBMENU_TEXT, prompts_submenu_keyboard())


@dp.callback_query(F.data == "menu_cover")
async def callback_menu_cover(callback: CallbackQuery):
    record_section_stat("cover")
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
    state["step"] = "text_choice"
    state.pop("photo_base64", None)
    await show_menu_screen(callback, COVER_TEXT_CHOICE_TEXT, cover_text_choice_keyboard())


@dp.callback_query(F.data == "cover_text_yes")
async def callback_cover_text_yes(callback: CallbackQuery):
    user_id = callback.from_user.id
    state = user_cover_state.get(user_id)
    if not state:
        await callback.answer("Сессия сброшена, начни заново через меню.", show_alert=True)
        return
    state["step"] = "awaiting_text"
    await show_menu_screen(callback, COVER_AWAITING_TEXT_TEXT, cover_awaiting_text_keyboard())


@dp.callback_query(F.data == "cover_text_no")
async def callback_cover_text_no(callback: CallbackQuery):
    user_id = callback.from_user.id
    state = user_cover_state.get(user_id)
    if not state:
        await callback.answer("Сессия сброшена, начни заново через меню.", show_alert=True)
        return
    state["step"] = "format"
    state.pop("cover_text", None)
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
    cover_text = state.get("cover_text")

    reply = await with_typing(chat_id, generate_cover_prompt(lyrics, photo_base64, ratio, cover_text))
    user_cover_state.pop(user_id, None)
    await send_prompt_reply(chat_id, reply, cover_result_keyboard())


@dp.callback_query(F.data.in_([f"prompt_{k}" for k in PROMPT_CONFIG.keys()]))
async def callback_prompt_select(callback: CallbackQuery):
    """Пользователь выбрал тему (Suno/Картинка/Видео) — показываем выбор версии/нейросети."""
    topic = callback.data.split("_", 1)[1]
    record_section_stat(f"prompt_{topic}")
    user_id = callback.from_user.id
    user_prompt_mode[user_id] = topic
    user_prompt_target.pop(user_id, None)
    user_video_frames.pop(user_id, None)
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
    user_video_frames.pop(user_id, None)

    intro = config["intro_after_target"].format(target=target_label)
    await show_menu_screen(callback, intro, prompt_active_keyboard())


@dp.callback_query(F.data == "menu_back")
async def callback_menu_back(callback: CallbackQuery):
    user_translate_mode.pop(callback.from_user.id, None)
    user_prompt_mode.pop(callback.from_user.id, None)
    user_prompt_target.pop(callback.from_user.id, None)
    user_cover_state.pop(callback.from_user.id, None)
    user_video_frames.pop(callback.from_user.id, None)
    await show_menu_screen(callback, MENU_MAIN_TEXT, main_menu_keyboard())


# ==================== ОСНОВНЫЕ КОМАНДЫ ====================

@dp.message(CommandStart())
async def cmd_start(message: Message):
    register_user(message.from_user.id)
    user_histories[message.from_user.id] = {}
    user_roles[message.from_user.id] = DEFAULT_ROLE
    save_user_roles()
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
    user_id = message.from_user.id
    role = user_roles.get(user_id, DEFAULT_ROLE)
    role_label = ROLE_CONFIG.get(role, ROLE_CONFIG[DEFAULT_ROLE])["label"]
    get_user_history(user_id, role).clear()
    save_histories()
    await message.answer(
        f"История для роли «{role_label}» очищена.", reply_markup=main_menu_button_keyboard()
    )


@dp.message(F.text == "/help")
async def cmd_help(message: Message):
    await message.answer(MENU_HELP_TEXT, parse_mode="HTML", reply_markup=main_menu_button_keyboard())


async def get_ai_reply(
    user_id: int,
    user_text: str,
    username: str | None = None,
    first_name: str | None = None,
    role: str = DEFAULT_ROLE,
) -> str:
    """Отправляет текст в Groq и возвращает ответ, обновляя историю диалога.
    role — id выбранной роли общения (см. ROLE_CONFIG); у каждой роли своя история,
    как отдельные вкладки на сайте."""
    role_config = ROLE_CONFIG.get(role, ROLE_CONFIG[DEFAULT_ROLE])
    system_prompt = role_config["system_prompt"]

    history = get_user_history(user_id, role)
    history.append({"role": "user", "content": user_text})
    trimmed_history = history[-20:]
    messages = [{"role": "system", "content": system_prompt}] + trimmed_history

    try:
        reply = clean_reply(chat_completion_with_fallback(messages, temperature=0.85, max_tokens=1024))
    except Exception as e:
        logging.exception("Ошибка при обращении к AI")
        reply = describe_groq_error(e)

    history.append({"role": "assistant", "content": reply})
    save_histories()

    await sync_message_to_site(user_id, username, first_name, "user", user_text, persona=role)
    await sync_message_to_site(user_id, username, first_name, "assistant", reply, persona=role)

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
            f"запросов в {RATE_LIMIT_WINDOW} секунд).",
            reply_markup=main_menu_button_keyboard(),
        )
        return

    # Убираем саму команду "/translate" из текста
    remainder = message.text[len("/translate"):].strip()

    if not remainder:
        await message.answer(
            "Использование:\n"
            "/translate <текст> — авто-перевод (RU→EN, другое→RU)\n"
            "/translate <код языка> <текст> — перевод на конкретный язык\n\n"
            "Например: /translate en Привет, как дела?",
            reply_markup=main_menu_button_keyboard(),
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
    await message.answer(f"🌐 {translation}", reply_markup=main_menu_button_keyboard())


# ==================== РАССЫЛКА (ТОЛЬКО ДЛЯ АДМИНА) ====================

def is_admin(user_id: int) -> bool:
    return ADMIN_ID is not None and user_id == ADMIN_ID


@dp.message(F.text == "/stats")
async def cmd_stats(message: Message):
    """Показывает статистику использования бота. Доступно только админу."""
    if not is_admin(message.from_user.id):
        return  # Не админ — молча игнорируем

    total_users = len(registered_users)
    messages_total = stats["messages_total"]

    today = date.today()
    messages_today = stats["messages_by_date"].get(today.isoformat(), 0)

    messages_week = 0
    for i in range(7):
        day = (today - timedelta(days=i)).isoformat()
        messages_week += stats["messages_by_date"].get(day, 0)

    sections_sorted = sorted(stats["sections"].items(), key=lambda x: x[1], reverse=True)

    section_labels = {
        "chat": "💬 Общение",
        "translate": "🌐 Переводчик",
        "prompts": "🎨 Промпты (открытие раздела)",
        "prompt_suno": "🎵 Промпты → Suno",
        "prompt_image": "🖼 Промпты → Картинка",
        "prompt_video": "🎬 Промпты → Видео",
        "cover": "🖼 Обложка трека",
        "favorites": "⭐ Избранное",
    }

    if sections_sorted:
        sections_text = "\n".join(
            f"  {section_labels.get(name, name)}: {count}" for name, count in sections_sorted
        )
    else:
        sections_text = "  Пока данных нет"

    text = (
        "📊 <b>Статистика бота</b>\n\n"
        f"👥 Пользователей всего: {total_users}\n\n"
        f"💬 Сообщений всего: {messages_total}\n"
        f"💬 Сообщений сегодня: {messages_today}\n"
        f"💬 Сообщений за 7 дней: {messages_week}\n\n"
        f"📋 Популярность разделов:\n{sections_text}"
    )

    await message.answer(text, parse_mode="HTML", reply_markup=main_menu_button_keyboard())


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


@dp.message(F.text.startswith("/announce"))
async def cmd_announce(message: Message):
    """
    Админ присылает краткое описание того, что обновилось — нейросеть оформляет это
    в красивый пост, показывает его на подтверждение и только после согласия рассылает всем.
    Формат: /announce добавили роли общения, смену пароля, мобильную версию сайта
    """
    if not is_admin(message.from_user.id):
        return

    raw_notes = message.text[len("/announce"):].strip()
    if not raw_notes:
        await message.answer(
            "Использование:\n"
            "/announce <кратко, что обновилось>\n\n"
            "Например:\n"
            "/announce добавили роли общения (друг, наставник и другие), смену пароля в аккаунте, "
            "мобильную версию сайта"
        )
        return

    await bot.send_chat_action(message.chat.id, "typing")
    announcement = await generate_announcement(raw_notes)
    user_pending_announcement[message.from_user.id] = {"raw": raw_notes, "text": announcement}

    await message.answer(
        "📢 Вот что получилось — рассылаем как есть, переписать ещё раз, или отменить?\n\n" + announcement,
        reply_markup=announce_confirm_keyboard(),
    )


@dp.callback_query(F.data == "announce_send")
async def callback_announce_send(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        return
    pending = user_pending_announcement.pop(callback.from_user.id, None)
    if not pending:
        await callback.answer("Черновик не найден — начни заново через /announce", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)

    text = pending["text"]
    await save_announcement_to_site(text)
    await bot.send_message(callback.from_user.id, f"Начинаю рассылку на {len(registered_users)} пользователей...")

    sent = 0
    failed = 0
    for user_id in list(registered_users):
        try:
            await bot.send_message(user_id, text)
            sent += 1
        except Exception as e:
            logging.warning(f"Не удалось отправить оповещение пользователю {user_id}: {e}")
            failed += 1
        await asyncio.sleep(0.05)

    await bot.send_message(callback.from_user.id, f"Готово! Успешно: {sent}, не удалось: {failed}")


@dp.callback_query(F.data == "announce_retry")
async def callback_announce_retry(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        return
    pending = user_pending_announcement.get(callback.from_user.id)
    if not pending:
        await callback.answer("Черновик не найден — начни заново через /announce", show_alert=True)
        return

    await callback.answer("Переписываю…")
    new_text = await generate_announcement(pending["raw"])
    user_pending_announcement[callback.from_user.id]["text"] = new_text
    await callback.message.edit_text(
        "📢 Вот что получилось — рассылаем как есть, переписать ещё раз, или отменить?\n\n" + new_text,
        reply_markup=announce_confirm_keyboard(),
    )


@dp.callback_query(F.data == "announce_cancel")
async def callback_announce_cancel(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        return
    user_pending_announcement.pop(callback.from_user.id, None)
    await callback.answer("Отменено")
    await callback.message.edit_reply_markup(reply_markup=None)


# ==================== ОБЫЧНЫЕ СООБЩЕНИЯ ====================

@dp.message(F.text)
async def handle_message(message: Message):
    register_user(message.from_user.id)
    record_message_stat()
    user_id = message.from_user.id

    # Если пользователь только что нажал "Написать комментарий" к посту галереи —
    # это сообщение и есть текст комментария
    pending_post_id = user_gallery_comment_target.pop(user_id, None)
    if pending_post_id is not None:
        await bot.send_chat_action(message.chat.id, "typing")
        result = await comment_on_gallery(
            user_id, message.from_user.username, message.from_user.first_name, pending_post_id, message.text
        )
        if result is None:
            await message.answer("Галерея временно недоступна — попробуй чуть позже.", reply_markup=main_menu_button_keyboard())
        elif result.get("status") == "approved":
            await message.answer("✅ Комментарий опубликован!", reply_markup=main_menu_button_keyboard())
        else:
            await message.answer(
                f"🚫 Комментарий отклонён модерацией: {result.get('reject_reason') or 'нарушение правил платформы'}",
                reply_markup=main_menu_button_keyboard(),
            )
        return

    # Если пользователь только что нажал "Написать сообщение" в общем чате —
    # это сообщение и есть текст для общего чата
    if user_id in user_publicchat_pending:
        user_publicchat_pending.discard(user_id)
        await bot.send_chat_action(message.chat.id, "typing")
        result = await send_public_chat(user_id, message.from_user.username, message.from_user.first_name, message.text)
        if result is None:
            await message.answer("Общий чат временно недоступен — попробуй чуть позже.", reply_markup=main_menu_button_keyboard())
        elif result.get("status") == "approved":
            await message.answer("✅ Сообщение опубликовано в общем чате!", reply_markup=main_menu_button_keyboard())
        else:
            await message.answer(
                f"🚫 Сообщение отклонено модерацией: {result.get('reject_reason') or 'нарушение правил платформы'}",
                reply_markup=main_menu_button_keyboard(),
            )
        return

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
        elif step == "awaiting_text":
            cover_state["cover_text"] = message.text
            cover_state["step"] = "format"
            await message.answer(COVER_FORMAT_TEXT, reply_markup=cover_format_keyboard())
        else:
            await message.answer("Выбери один из вариантов кнопками выше 👆", reply_markup=main_menu_button_keyboard())
        return

    # Если у пользователя включён режим перевода — обрабатываем текст иначе
    mode = user_translate_mode.get(user_id)
    if mode is not None:
        if is_rate_limited(user_id):
            await message.answer(
                f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
                f"запросов в {RATE_LIMIT_WINDOW} секунд).",
                reply_markup=main_menu_button_keyboard(),
            )
            return

        # Пользователь вводит название/код языка после нажатия "Указать язык" — разрешено всегда
        if mode.get("awaiting_custom_lang"):
            custom_lang = message.text.strip()
            user_translate_mode[user_id] = {"target_lang": custom_lang}
            await message.answer(
                f"✅ Режим перевода на «{custom_lang}» включён.\n\nПросто присылай текст — переведу.",
                reply_markup=translate_active_keyboard(),
            )
            return

        # Если выбран режим "перевод голосовых" — просим прислать войс вместо текста
        if user_translate_input_type.get(user_id, "text") == "voice":
            await message.answer(
                "🎙 Сейчас включён режим перевода голосовых сообщений. Пришли войс — или переключись "
                "на «⌨️ Текста» в меню переводчика.",
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
                f"запросов в {RATE_LIMIT_WINDOW} секунд).",
                reply_markup=main_menu_button_keyboard(),
            )
            return

        reply = await with_typing(message.chat.id, get_prompt_reply(user_id, prompt_type, message.text))
        await send_prompt_reply(message.chat.id, reply, prompt_active_keyboard())
        return

    if is_rate_limited(user_id):
        await message.answer(
            f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
            f"запросов в {RATE_LIMIT_WINDOW} секунд).",
            reply_markup=main_menu_button_keyboard(),
        )
        return

    await bot.send_chat_action(message.chat.id, "typing")
    reply = await get_ai_reply(
        user_id,
        message.text,
        message.from_user.username,
        message.from_user.first_name,
        role=user_roles.get(user_id, DEFAULT_ROLE),
    )
    await message.answer(reply, reply_markup=main_menu_button_keyboard())

    if user_tts_enabled.get(user_id, False):
        await send_voice_reply(message.chat.id, reply)


@dp.message(F.voice)
async def handle_voice(message: Message):
    register_user(message.from_user.id)
    record_message_stat()
    user_id = message.from_user.id

    if is_rate_limited(user_id):
        await message.answer(
            f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
            f"запросов в {RATE_LIMIT_WINDOW} секунд).",
            reply_markup=main_menu_button_keyboard(),
        )
        return

    # Если у пользователя открыт переводчик, но выбран режим "текста" — голос не подходит
    translate_mode = user_translate_mode.get(user_id)
    if translate_mode is not None:
        if translate_mode.get("awaiting_custom_lang"):
            await message.answer(
                "⌨️ Напиши язык текстом — например: испанский, es, japanese.",
                reply_markup=translate_active_keyboard(),
            )
            return
        if user_translate_input_type.get(user_id, "text") == "text":
            await message.answer(
                "⌨️ Сейчас включён режим перевода текста. Напиши текст — или переключись "
                "на «🎙 Голосовых» в меню переводчика.",
                reply_markup=translate_active_keyboard(),
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
        await message.answer(f"Не удалось распознать голосовое сообщение: {e}", reply_markup=main_menu_button_keyboard())
        return
    finally:
        os.remove(tmp_path)

    if not recognized_text.strip():
        await message.answer("Не удалось разобрать речь, попробуй ещё раз.", reply_markup=main_menu_button_keyboard())
        return

    # Режим перевода голосовых — переводим распознанный текст и остаёмся в режиме
    if translate_mode is not None:
        translation = await with_typing(
            message.chat.id, translate_text(recognized_text, translate_mode.get("target_lang"))
        )
        await message.answer(
            f"🎤 Я услышал: «{recognized_text}»\n\n🌐 {translation}", reply_markup=translate_active_keyboard()
        )
        await send_voice_reply(message.chat.id, translation)
        return

    reply = await get_ai_reply(
        user_id,
        recognized_text,
        message.from_user.username,
        message.from_user.first_name,
        role=user_roles.get(user_id, DEFAULT_ROLE),
    )
    await message.answer(f"🎤 Я услышал: «{recognized_text}»\n\n{reply}", reply_markup=main_menu_button_keyboard())

    if user_tts_enabled.get(user_id, False):
        await send_voice_reply(message.chat.id, reply)


@dp.message(F.photo)
async def handle_photo(message: Message):
    register_user(message.from_user.id)
    record_message_stat()

    if is_rate_limited(message.from_user.id):
        await message.answer(
            f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_COUNT} "
            f"запросов в {RATE_LIMIT_WINDOW} секунд).",
            reply_markup=main_menu_button_keyboard(),
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
        cover_state["step"] = "text_choice"
        await message.answer(COVER_TEXT_CHOICE_TEXT, reply_markup=cover_text_choice_keyboard())
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

    # Если у пользователя включён режим составления промпта для видео —
    # первое фото сохраняем как первый кадр, второе — как последний, и составляем промпт
    if user_prompt_mode.get(message.from_user.id) == "video":
        user_id = message.from_user.id
        frames = user_video_frames.setdefault(user_id, {"first": None, "last": None, "note": ""})
        if message.caption:
            frames["note"] = (frames["note"] + " " + message.caption).strip()

        if frames["first"] is None:
            frames["first"] = image_base64
            await message.answer(
                "🎬 Сохранил как первый кадр!\n\n"
                "Можешь прислать ещё и последний кадр сцены — или просто опиши словами, "
                "что происходит между ними, и я составлю промпт.",
                reply_markup=prompt_active_keyboard(),
            )
            return

        frames["last"] = image_base64
        reply = await with_typing(
            message.chat.id,
            get_video_prompt_from_frames(user_id, frames["note"], frames["first"], frames["last"]),
        )
        user_video_frames.pop(user_id, None)
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
    role = user_roles.get(user_id, DEFAULT_ROLE)
    history = get_user_history(user_id, role)
    history.append({"role": "user", "content": f"[Отправил фото] {question}"})
    history.append({"role": "assistant", "content": reply})
    save_histories()

    await sync_message_to_site(
        user_id, message.from_user.username, message.from_user.first_name,
        "user", f"[Отправил фото] {question}", persona=role,
    )
    await sync_message_to_site(
        user_id, message.from_user.username, message.from_user.first_name, "assistant", reply, persona=role
    )

    await message.answer(reply, reply_markup=main_menu_button_keyboard())


async def main():
    print("Бот запущен. Не закрывай это окно.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
