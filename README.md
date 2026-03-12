# RealTimePIIAnonymizer

Нужна версия Python 3.11 или 3.12.

## Установка

```bash
pip install -r requirements.txt
```

## PII-маскирование

Файл `masker.py` содержит:
- `mask_pii(text)` - совместимую функцию, возвращающую только замаскированный текст;
- `mask_pii_with_mapping(text)` - возвращает `masked_text` и mapping для обратной подстановки;
- `unmask_pii(masked_text, mapping)` - восстанавливает исходный текст;
- `save_mapping(...)` / `load_mapping(...)` - сохранение и загрузка mapping в JSON.

Для повторяющихся сущностей используются уникальные плейсхолдеры, например: `[EMAIL_1]`, `[EMAIL_2]`, `[NAME_1]`.

## Web чат с LLM

Реализован минимальный интерфейс в темной теме с белыми скругленными контурами:
- backend: `app/main.py` (FastAPI);
- клиент LLM: `agent/agent.py` (Mistral);
- frontend: `app/static/index.html`, `app/static/styles.css`, `app/static/app.js`.

Фронтенд использует потоковый endpoint `POST /chat/stream`, поэтому ответ LLM появляется постепенно по мере генерации.

### Настройка ключа

1. Скопируйте `.env.example` в `.env`.
2. Укажите `MISTRAL_API_KEY`.

### Запуск

```bash
uvicorn app.main:app --reload
```

После запуска откройте `http://127.0.0.1:8000`.


[Ссылка на задание](https://event.cu.ru/master-hackathon-ml2026?utm_source=ooh&utm_medium=smm.unp&utm_campaign=cu.events.master%2Fhackathon-ML&utm_term=itmo.master)
