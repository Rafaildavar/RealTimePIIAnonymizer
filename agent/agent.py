import os
from typing import Iterator, Optional

from mistralai.client import Mistral


class MissingApiKeyError(ValueError):
    '''Нет API-ключа в окружении.'''


class InvalidApiKeyError(RuntimeError):
    '''Провайдер отклонил API-ключ (обычно 401).'''


class MistralAgent:
    '''
    Минимальный клиент для запроса к Mistral Chat API.

    Ключ читается из переменной окружения MISTRAL_API_KEY,
    модель по умолчанию - mistral-medium-latest.
    '''

    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-medium-latest"):
        self.api_key = (api_key or os.getenv("MISTRAL_API_KEY") or "").strip()
        if not self.api_key:
            raise MissingApiKeyError("MISTRAL_API_KEY is not set")

        self.model = model
        self.client = Mistral(api_key=self.api_key)

    def _raise_auth_error_if_needed(self, exc: Exception) -> None:
        '''Преобразует ошибки авторизации провайдера в понятное доменное исключение.'''
        message = str(exc)
        if "401" in message or "Unauthorized" in message:
            raise InvalidApiKeyError(
                "MISTRAL_API_KEY is invalid or expired. Regenerate key and restart server."
            ) from exc

    def _extract_stream_text(self, chunk: object) -> str:
        '''Достает текст из чанка стрима Mistral SDK в нескольких возможных форматах.'''
        payload = getattr(chunk, "data", chunk)
        choices = getattr(payload, "choices", None)
        if not choices:
            return ""

        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts)

        return ""

    def ask(self, prompt: str) -> str:
        '''Возвращает текстовый ответ модели на пользовательский prompt.'''
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty")

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "system", "content": "Ты помощник по ответам на банковские вопросы. Отвечай кратко и по существу. Сейчас можешь выдумывать информацию. Ты получаешь вопрос пользователя без персональных данных, они были скрыты тегами в []. При ответе используй эти теги и никогда не меняй информацию внутри []. Если в вопросе нет тегов, просто ответь на него."},
                    {"role": "user", "content": prompt}],
            )
        except Exception as exc:
            self._raise_auth_error_if_needed(exc)
            raise

        content = response.choices[0].message.content
        if isinstance(content, str):
            return content

        # На случай, если SDK вернет контент как список блоков.
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
            if parts:
                return "".join(parts)

        return str(content)

    def stream_answer(self, prompt: str) -> Iterator[str]:
        '''Стримит ответ модели по мере генерации текста.'''
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty")

        try:
            stream = self.client.chat.stream(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            self._raise_auth_error_if_needed(exc)
            raise

        try:
            for chunk in stream:
                text = self._extract_stream_text(chunk)
                if text:
                    yield text
        except Exception as exc:
            self._raise_auth_error_if_needed(exc)
            raise


if __name__ == '__main__':
    agent = MistralAgent()
    print("".join(agent.stream_answer("What is the best French cheese?")))
