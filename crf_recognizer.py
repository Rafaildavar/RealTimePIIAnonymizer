import re
import joblib
from dataclasses import dataclass
from typing import List, Dict, Optional, Set


# =========================================================
# 1. Токенизация
# =========================================================

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

@dataclass
class EntityMatch:
    entity_type: str
    start: int
    end: int
    text: str
    score: float
    source: str

    @property
    def length(self) -> int:
        return self.end - self.start
@dataclass
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> List[Token]:
    return [Token(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


# =========================================================
# 2. Словари для fallback-логики
# =========================================================

FIRST_NAMES = {
    "иван", "петр", "сергей", "анна", "мария", "елена", "дмитрий",
    "алексей", "андрей", "максим", "наталья", "екатерина", "виктория",
    "михаил", "эмиль", "артем", "артём", "никита", "кирилл", "илья",
    "роман", "даниил", "денис", "егор"
}

LAST_NAMES = {
    "иванов", "петров", "сидоров", "кузнецов", "кузнецова", "смирнов",
    "смирнова", "волков", "соколов", "попов", "федоров", "фёдоров",
    "орлов", "сергеев", "алексеев", "ахметов", "романов", "егоров"
}


# =========================================================
# 3. Признаки для CRF
# Важно: они должны быть совместимы с тем, как обучалась модель
# =========================================================

def token2features(tokens: List[Token], i: int) -> Dict[str, object]:
    token = tokens[i].text
    token_lower = token.lower()

    features = {
        "bias": 1.0,
        "word.lower()": token_lower,
        "word[-3:]": token[-3:],
        "word[-2:]": token[-2:],
        "word.isupper()": token.isupper(),
        "word.istitle()": token.istitle(),
        "word.isdigit()": token.isdigit(),
        "word.hasdigit()": any(ch.isdigit() for ch in token),
        "word.hasalpha()": any(ch.isalpha() for ch in token),
        "word.has_at()": "@" in token,
        "word.has_dash()": "-" in token,
        "word.has_dot()": "." in token,
        "len": len(token),
        "is_capitalized_rus": bool(re.match(r"^[А-ЯЁ][а-яё]+$", token)),
        "is_first_name_dict": token_lower in FIRST_NAMES,
        "is_last_name_dict": token_lower in LAST_NAMES,
        "is_name_marker": token_lower in {"зовут", "имя", "клиент", "это", "данные"},
        "is_address_marker": token_lower in {
            "адрес", "город", "г", "улица", "ул", "ул.",
            "проспект", "пр", "пр.", "дом", "д", "д.",
            "квартира", "кв", "кв.", "невский", "ленина",
            "тверская", "баумана"
        },
    }

    if i > 0:
        prev_token = tokens[i - 1].text
        prev_lower = prev_token.lower()
        features["-1:word.lower()"] = prev_lower
        features["-1:word.istitle()"] = prev_token.istitle()
        features["-1:is_name_marker"] = prev_lower in {"зовут", "это", "клиент", "я"}
        features["-1:is_address_marker"] = prev_lower in {
            "адрес", "город", "г", "улица", "ул", "ул.",
            "проспект", "дом", "д", "квартира", "кв"
        }
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        next_token = tokens[i + 1].text
        next_lower = next_token.lower()
        features["+1:word.lower()"] = next_lower
        features["+1:word.istitle()"] = next_token.istitle()
        features["+1:is_address_marker"] = next_lower in {
            "улица", "ул", "ул.", "проспект", "дом", "д", "квартира", "кв"
        }
    else:
        features["EOS"] = True

    return features


def sent2features(tokens: List[Token]) -> List[Dict[str, object]]:
    return [token2features(tokens, i) for i in range(len(tokens))]


# =========================================================
# 4. Вспомогательные fallback-правила
# Используются как усиление для NAME и ADDRESS
# =========================================================

def fallback_name_detection(text: str):
    matches = []

    patterns = [
        re.compile(r"\b(?:меня зовут)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)"),
        re.compile(r"\b(?:это)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)"),
        re.compile(r"\b(?:я)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)"),
        re.compile(r"\b([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)\b"),
    ]

    for pattern in patterns:
        for m in pattern.finditer(text):
            candidate = m.group(1).strip()
            parts = candidate.split()
            lowered = [p.lower() for p in parts]

            score = 0.70
            if any(p in FIRST_NAMES for p in lowered):
                score += 0.10
            if any(p in LAST_NAMES for p in lowered):
                score += 0.10

            if len(parts) == 1 and lowered[0] not in FIRST_NAMES:
                continue

            matches.append({
                "entity_type": "NAME",
                "start": m.start(1),
                "end": m.end(1),
                "text": candidate,
                "score": min(score, 0.90),
                "source": "crf_fallback_name",
            })

    return matches


def fallback_address_detection(text: str):
    matches = []

    patterns = [
        re.compile(
            r"\b(?:адрес(?: проживания)?|живу по адресу)\s+([А-ЯЁA-Z][^,.!\n]{5,80})",
            re.IGNORECASE
        ),
        re.compile(
            r"\b([А-ЯЁA-Z][а-яёa-z-]+(?:\s+[А-ЯЁA-Z][а-яёa-z-]+)?\s+(?:улица|ул\.?|проспект|пр\.?|дом|д\.?)\s+[^,.!\n]{1,40})",
            re.IGNORECASE
        ),
    ]

    for pattern in patterns:
        for m in pattern.finditer(text):
            candidate = m.group(1).strip()
            matches.append({
                "entity_type": "ADDRESS",
                "start": m.start(1),
                "end": m.end(1),
                "text": candidate,
                "score": 0.78,
                "source": "crf_fallback_address",
            })

    return matches


# =========================================================
# 5. Основной recognizer
# ВАЖНО:
# EntityMatch должен существовать в основном проекте
# Если IDE ругается, импортируйте его из вашего основного файла
# =========================================================

class CRFRecognizer:
    name = "crf_recognizer"

    def __init__(self, model_path: str, allowed_types: Optional[Set[str]] = None):
        self.model = joblib.load(model_path)
        self.allowed_types = set(allowed_types or {"NAME", "ADDRESS"})

    def _bio_to_matches(self, text: str, tokens: List[Token], labels: List[str]):
        matches = []
        current_type = None
        start_idx = None
        end_idx = None

        for i, label in enumerate(labels):
            if label == "O":
                if current_type is not None and current_type in self.allowed_types:
                    matches.append(
                        EntityMatch(
                            entity_type=current_type,
                            start=tokens[start_idx].start,
                            end=tokens[end_idx].end,
                            text=text[tokens[start_idx].start:tokens[end_idx].end],
                            score=0.80,
                            source=self.name,
                        )
                    )
                    current_type = None
                    start_idx = None
                    end_idx = None
                continue

            if "-" not in label:
                continue

            prefix, entity_type = label.split("-", 1)

            if prefix == "B":
                if current_type is not None and current_type in self.allowed_types:
                    matches.append(
                        EntityMatch(
                            entity_type=current_type,
                            start=tokens[start_idx].start,
                            end=tokens[end_idx].end,
                            text=text[tokens[start_idx].start:tokens[end_idx].end],
                            score=0.80,
                            source=self.name,
                        )
                    )
                current_type = entity_type
                start_idx = i
                end_idx = i

            elif prefix == "I":
                if current_type == entity_type:
                    end_idx = i
                else:
                    current_type = entity_type
                    start_idx = i
                    end_idx = i

        if current_type is not None and current_type in self.allowed_types:
            matches.append(
                EntityMatch(
                    entity_type=current_type,
                    start=tokens[start_idx].start,
                    end=tokens[end_idx].end,
                    text=text[tokens[start_idx].start:tokens[end_idx].end],
                    score=0.80,
                    source=self.name,
                )
            )

        return matches

    def _convert_fallback_matches(self, raw_matches):
        result = []
        for m in raw_matches:
            if m["entity_type"] in self.allowed_types:
                result.append(
                    EntityMatch(
                        entity_type=m["entity_type"],
                        start=m["start"],
                        end=m["end"],
                        text=m["text"],
                        score=m["score"],
                        source=m["source"],
                    )
                )
        return result

    def find(self, text: str):
        tokens = tokenize(text)
        if not tokens:
            return []

        X = [sent2features(tokens)]
        labels = self.model.predict(X)[0]

        crf_matches = self._bio_to_matches(text, tokens, labels)

        fallback_matches = []
        if "NAME" in self.allowed_types:
            fallback_matches.extend(self._convert_fallback_matches(fallback_name_detection(text)))
        if "ADDRESS" in self.allowed_types:
            fallback_matches.extend(self._convert_fallback_matches(fallback_address_detection(text)))

        return crf_matches + fallback_matches