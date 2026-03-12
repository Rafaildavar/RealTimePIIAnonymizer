import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

try:
    from natasha import (
        Segmenter,
        NewsEmbedding,
        NewsNERTagger,
        Doc,
    )
except Exception:
    Segmenter = NewsEmbedding = NewsNERTagger = Doc = None


RULE_PATTERNS = [
    (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+7|8|7)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-()]*\d{2}[\s\-()]*\d{2}\b"), "[PHONE]"),
    (re.compile(r"\b\d{4}[\s-]?\d{6}\b"), "[PASSPORT]"),
    (re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"), "[CARD]"),
]

ENTITY_TAGS = {
    "PER": "[NAME]",
    "LOC": "[ADDRESS]",
}


@dataclass(frozen=True)
class MaskResult:
    '''
    Результат маскирования: текст с плейсхолдерами и словарь для обратной подстановки.
    mapping хранится в формате {"[TAG_n]": "исходное значение"}.
    '''
    masked_text: str
    mapping: Dict[str, str]


def _normalize_entity(tag: str) -> str:
    '''Преобразует тег вида "[EMAIL]" в имя сущности "EMAIL".'''
    return tag.strip("[]")


def _make_placeholder(entity: str, counters: Dict[str, int]) -> str:
    '''
    Генерирует уникальный плейсхолдер для сущности и увеличивает счетчик.
    Пример: для entity="NAME" вернет "[NAME_1]", затем "[NAME_2]".
    '''
    counters[entity] = counters.get(entity, 0) + 1
    return f"[{entity}_{counters[entity]}]"


@lru_cache(maxsize=1)
def _get_natasha_components() -> Optional[Tuple[object, object]]:
    """Инициализирует Natasha-компоненты один раз и переиспользует их из кэша."""
    if Doc is None:
        return None

    try:
        segmenter = Segmenter()
        emb = NewsEmbedding()
        ner_tagger = NewsNERTagger(emb)
        return segmenter, ner_tagger
    except Exception:
        return None


def _mask_with_rules(text: str, counters: Dict[str, int], mapping: Dict[str, str]) -> str:
    """Маскирует PII по regex-правилам и сохраняет соответствия для восстановления."""
    masked = text

    for pattern, replacement in RULE_PATTERNS:
        entity = _normalize_entity(replacement)

        def _repl(match: re.Match) -> str:
            placeholder = _make_placeholder(entity, counters)
            mapping[placeholder] = match.group(0)
            return placeholder

        masked = pattern.sub(_repl, masked)

    return masked


def mask_pii_with_mapping(text: str) -> MaskResult:
    """Возвращает замаскированный текст и mapping для обратной подстановки."""
    counters: Dict[str, int] = {}
    mapping: Dict[str, str] = {}

    masked = _mask_with_rules(text, counters=counters, mapping=mapping)

    components = _get_natasha_components()
    if components is None:
        return MaskResult(masked_text=masked, mapping=mapping)

    segmenter, ner_tagger = components
    doc = Doc(masked)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    # Идем справа налево, чтобы индексы span не "съезжали" после замены.
    for span in reversed(doc.spans):
        replacement = ENTITY_TAGS.get(span.type)
        if not replacement:
            continue

        entity = _normalize_entity(replacement)
        placeholder = _make_placeholder(entity, counters)
        original = masked[span.start:span.stop]

        mapping[placeholder] = original
        masked = masked[:span.start] + placeholder + masked[span.stop:]

    return MaskResult(masked_text=masked, mapping=mapping)


def mask_pii(text: str) -> str:
    """Совместимая обертка: возвращает только замаскированный текст."""
    return mask_pii_with_mapping(text).masked_text


def save_mapping(mapping: Dict[str, str], file_path: str) -> None:
    """Сохраняет mapping в JSON-файл."""
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)


def load_mapping(file_path: str) -> Dict[str, str]:
    """Загружает mapping из JSON-файла."""
    with open(file_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict):
        raise ValueError("Mapping JSON должен быть объектом вида {placeholder: original}")

    return {str(k): str(v) for k, v in payload.items()}


if __name__ == "__main__":
    text = "Я Анна Ивановна, email anna@example.com, второй email anna@example.com"

    result = mask_pii_with_mapping(text)
    print("masked:", result.masked_text)
    print("mapping:", result.mapping)
    print("restored:", unmask_pii(result.masked_text, result.mapping))
