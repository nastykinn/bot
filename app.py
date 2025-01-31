import json
import requests
import re
import feedparser
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
import wikipediaapi
import asyncio

# Инициализация FastAPI приложения
app = FastAPI()

# Используем модель distilGPT-2 для экономии памяти
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Ваши API-ключи для Google Custom Search
GOOGLE_API_KEY = "AIzaSyBdnfF8BSKw44qdfh05bRPW_gfJyu9LEB4"
GOOGLE_CSE_ID = "8194a0485f01e4051"

# URL для RSS-лент и сайтов, которые будем использовать для получения новостей и данных
RSS_FEEDS = [
    "https://news.itmo.ru/ru/rss/",  # RSS лента новостей ИТМО
    "https://www.itmo.ru/ru/news/feed.xml",  # Дополнительная лента новостей ИТМО
    "https://museum.itmo.ru/",  # Сайт музея ИТМО
    "https://itmo.events/",  # Сайт событий ИТМО
    "https://vk.com/itmoru",  # Официальная страница ВКонтакте
    "https://en.wikipedia.org/wiki/ITMO_University",  # Статья в Википедии о ИТМО
    "https://itmo.vercel.app/"  # Страница на Vercel
]

# Используем wikipedia-api для извлечения текста из Википедии
wiki_wiki = wikipediaapi.Wikipedia(
    language='ru',
    user_agent='YourAppName/1.0 (contact@yourapp.com)'  # Замените на свое имя или контактный email
)


# Модели для запроса и ответа
class QueryRequest(BaseModel):
    query: str
    id: int


class QueryResponse(BaseModel):
    id: int
    answer: Optional[int]  # Заменили int | None на Optional[int]
    reasoning: str
    sources: list
    latest_news: list

def get_wikipedia_summary(query):
    """
    Извлекает резюме из статьи Википедии.
    """
    page = wiki_wiki.page(query)
    if page.exists():
        return page.text.split('\n')[0]  # Возвращаем только первый абзац
    else:
        return None


async def get_sources_from_google(query):
    """
    Асинхронно выполняет поиск через Google Custom Search API, используя сайты из вашей CSE.
    """
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"

    try:
        response = await asyncio.to_thread(requests.get, search_url)
        response.raise_for_status()
        search_results = response.json()

        if 'items' in search_results:
            sources = [item['link'] for item in search_results.get('items', [])[:3]]  # Ограничиваем 3 ссылками
            return sources

        return []  # Если результатов нет, возвращаем пустой список

    except requests.exceptions.RequestException as e:
        return []


async def scrape_page_text(url):
    """
    Асинхронно загружает страницу и извлекает чистый текст без HTML-кода.
    """
    try:
        response = await asyncio.to_thread(requests.get, url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Удаляем ненужные элементы (меню, скрипты и т. д.)
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.extract()

        # Извлекаем чистый текст
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()

        # Применяем простую фильтрацию, чтобы отфильтровать ненужную информацию
        if len(text.split()) < 50:  # Если текст слишком короткий, он маловероятно полезен
            return None

        return text[:2000]  # Ограничиваем текст, чтобы не перегружать модель

    except requests.exceptions.RequestException:
        return None


async def get_model_response(query, context=None):
    """
    Асинхронно генерирует ответ на основе информации с сайта.
    """
    if context:  # Если есть контекст, передаем его
        query = f"{query}\n\nКонтекст: {context}"

    # Ограничиваем длину входных данных (max_length)
    max_input_length = 512  # Ограничение длины ввода
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=max_input_length)

    # Генерация ответа (max_new_tokens вместо max_length)
    outputs = await asyncio.to_thread(model.generate, inputs["input_ids"],
                                      max_new_tokens=150)  # Генерация более длинного текста

    # Декодируем результат
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return clean_response(response)


def clean_response(response):
    """
    Очищает ответ:
    - Убирает лишние фразы
    - Оставляет только нужный ответ
    """
    response = response.strip()
    response = re.sub(r'\n+', ' ', response)  # Убираем лишние переводы строк

    # Убираем ненужные вводные фразы
    response = re.sub(r"(Ответ:|я не могу.*?говорить).*", "", response, flags=re.IGNORECASE)

    # Оставляем только первые предложения
    sentences = re.split(r'(?<=[.!?])\s+', response)
    short_response = " ".join(sentences[:2])

    # Если ответ пустой или мусорный, заменяем на "Ответ не найден"
    if len(short_response) < 5 or "не знаю" in short_response.lower():
        return "Ответ не найден."

    return short_response


async def extract_correct_answer(query, model_response):
    """
    Асинхронно определяет правильный вариант ответа (1-10), если он есть в тексте.
    """
    if any(str(i) in query for i in range(1, 11)):  # Проверяем, есть ли варианты 1-10
        match = re.search(r"\b([1-9]|10)\b", model_response)
        if match:
            return int(match.group(1))
    return None  # Если в вопросе нет вариантов, `answer=null`


async def get_rss_news():
    """
    Асинхронно получаем последние новости из RSS-лент.
    """
    all_news = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:3]:  # Ограничиваем до 3 последних новостей
            all_news.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published
            })
    return all_news


async def handle_request(query, request_id):
    sources = await get_sources_from_google(query)  # Поиск ссылок через Google API

    # Парсим страницы (если есть)
    context = None
    if sources:
        for source in sources:
            context = await scrape_page_text(source)
            if context:
                break  # Если текст найден, используем его

    # Пытаемся извлечь резюме из Википедии
    if query.lower() == "университет итмо" or "itmo university":
        context = get_wikipedia_summary("ITMO_University")  # Извлекаем текст из Википедии

    # Генерация ответа с учетом контекста
    model_answer = await get_model_response(query, context)

    # Определяем номер правильного ответа (если есть варианты 1-10)
    answer = await extract_correct_answer(query, model_answer)

    # Получаем новости, но исключаем их из ответа
    # news = await get_rss_news()  # Если новости не нужны, закомментируйте эту строку

    # Возвращаем JSON без поля "latest_news"
    response = {
        "id": request_id,
        "answer": answer if answer else None,
        "reasoning": model_answer,
        "sources": sources if sources else [],
        # "latest_news": news  # Убираем поле с новостями из ответа
    }

    return json.dumps(response, ensure_ascii=False, indent=2)


@app.post("/api/request")
async def api_request(data: QueryRequest):
    try:
        response_json = await handle_request(data.query, data.id)
        return JSONResponse(content=json.loads(response_json))  # Отправляем JSON-ответ
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")


# Запуск сервера
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
