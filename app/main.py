from pathlib import Path
from typing import Iterable, Iterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.agent import InvalidApiKeyError, MissingApiKeyError, MistralAgent
from masker import mask_pii_with_mapping
from unmask import unmask_pii

import sys
from loguru import logger

logger.remove()
# stdout обычно виден стабильнее в терминале с uvicorn --reload
logger.add(sys.stdout, level="INFO")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


app = FastAPI(title="LLM Chat")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


def _validate_message(message: str) -> str:
    text = message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message must not be empty")
    return text


def _build_agent() -> MistralAgent:
    try:
        return MistralAgent()
    except MissingApiKeyError as exc:
        raise HTTPException(
            status_code=500,
            detail="MISTRAL_API_KEY is missing. Put it into .env and restart server.",
        ) from exc
    except InvalidApiKeyError as exc:
        raise HTTPException(
            status_code=401,
            detail="MISTRAL_API_KEY is invalid or expired. Update .env and restart server.",
        ) from exc


def _iter_unmasked_chunks(chunks: Iterable[str], mapping: dict[str, str]) -> Iterator[str]:
    """Размаскирует поток, удерживая хвост с незакрытым '[' до прихода ']' ."""
    pending = ""

    for chunk in chunks:
        logger.info(f"raw chunk: {chunk!r}")
        pending += chunk

        last_open = pending.rfind("[")
        last_close = pending.rfind("]")

        # Если в хвосте есть незакрытая скобка, держим этот фрагмент до следующего чанка.
        if last_open > last_close:
            ready = pending[:last_open]
            pending = pending[last_open:]
        else:
            ready = pending
            pending = ""

        if ready:
            safe = unmask_pii(ready, mapping)
            logger.info(f"emitted chunk: {safe!r}")
            yield safe

    if pending:
        safe_tail = unmask_pii(pending, mapping)
        logger.info(f"emitted tail: {safe_tail!r}")
        yield safe_tail


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    text = _validate_message(payload.message)
    agent = _build_agent()
    mask_result = mask_pii_with_mapping(text)

    logger.info(f"/chat masked input: {mask_result.masked_text}")

    try:
        masked_answer = agent.ask(mask_result.masked_text)
        answer = unmask_pii(masked_answer, mask_result.mapping)
        logger.info(f"/chat final answer: {answer!r}")
        return ChatResponse(answer=answer)
    except InvalidApiKeyError as exc:
        logger.exception("/chat invalid api key")
        raise HTTPException(
            status_code=401,
            detail="MISTRAL_API_KEY is invalid or expired. Update .env and restart server.",
        ) from exc
    except ValueError as exc:
        logger.exception("/chat bad request")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("/chat llm failed")
        raise HTTPException(status_code=500, detail=f"LLM request failed: {exc}") from exc


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    text = _validate_message(payload.message)
    agent = _build_agent()
    mask_result = mask_pii_with_mapping(text)

    logger.info(f"/chat/stream masked input: {mask_result.masked_text}")

    def generate() -> Iterator[str]:
        try:
            stream = agent.stream_answer(mask_result.masked_text)
            for safe_chunk in _iter_unmasked_chunks(stream, mask_result.mapping):
                yield safe_chunk
        except InvalidApiKeyError:
            logger.exception("/chat/stream invalid api key")
            yield "\n[Ошибка: MISTRAL_API_KEY is invalid or expired. Update .env and restart server.]"
        except Exception as exc:
            logger.exception("/chat/stream failed")
            yield f"\n[Ошибка стрима: {exc}]"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
