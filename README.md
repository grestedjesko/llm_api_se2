# Local LLM API

FastAPI сервер для локального запуска LLM (Large Language Models) с OpenAI-подобным API.

### Установка зависимостей

```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Запуск сервера

```bash
# Используйте uvicorn из виртуального окружения
./venv/bin/uvicorn api:app --reload

# Или активируйте venv и используйте обычную команду
source venv/bin/activate
uvicorn api:app --reload
```

Сервер будет доступен по адресу: http://127.0.0.1:8000

### API документация

После запуска сервера, документация доступна по адресу:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Конфигурация

Настройки хранятся в `config.py` и могут быть переопределены через переменные окружения:

```bash
MODEL_NAME=Qwen/Qwen3-4B  # модель по умолчанию
```

## Тестирование

### Unit-тесты (быстрые, для CI)

```bash
pytest -v -m "not integration"
```

### Интеграционные тесты (с реальной моделью)

```bash
pytest -v -m integration -s
```

## Эндпоинты

### GET /health

Проверка состояния сервера.

```bash
curl http://localhost:8000/health
```

### POST /completions/create

Генерация текста (OpenAI-style API).

```bash
curl -X POST http://localhost:8000/completions/create \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Требования

- Python 3.11+
- 8GB+ RAM (для загрузки модели)
- GPU с поддержкой CUDA или Apple Silicon (MPS) рекомендуется для быстрой генерации

## Лицензия

MIT

