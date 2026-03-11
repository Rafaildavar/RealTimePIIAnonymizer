import re
from functools import lru_cache

from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsNERTagger,
    Doc,
)


RULE_PATTERNS = [
    (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-()]*\d{2}[\s\-()]*\d{2}\b"), "[PHONE]"),
    (re.compile(r"\b\d{4}[\s-]?\d{6}\b"), "[PASSPORT]"),
    (re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"), "[CARD]"),
]

ENTITY_TAGS = {
    "PER": "[NAME]",
    "LOC": "[ADDRESS]",
}


@lru_cache(maxsize=1)
def _get_natasha_components():
    try:
        segmenter = Segmenter()
        emb = NewsEmbedding()
        ner_tagger = NewsNERTagger(emb)
        return segmenter, ner_tagger
    except Exception:
        return None


def _mask_with_rules(text: str) -> str:
    masked = text
    for pattern, replacement in RULE_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked


def mask_pii(text: str) -> str:
    masked = _mask_with_rules(text)
    components = _get_natasha_components()
    if components is None:
        return masked

    segmenter, ner_tagger = components
    doc = Doc(masked)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    for span in reversed(doc.spans):
        replacement = ENTITY_TAGS.get(span.type)
        if replacement:
            masked = masked[:span.start] + replacement + masked[span.stop:]

    return masked


if __name__ == "__main__":
    text = "Я Иван Иванов, мой email test@example.com, а карта 4276 1234 5678 9012"
    print(mask_pii(text))
