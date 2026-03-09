import re
import joblib
from dataclasses import dataclass
from typing import List, Dict

MODEL_PATH = "crf_pii_model.joblib"
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Token:
    text: str
    start: int
    end: int


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


def tokenize(text: str) -> List[Token]:
    return [Token(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def token2features(tokens: List[Token], i: int) -> Dict[str, object]:
    token = tokens[i].text

    features = {
        "bias": 1.0,
        "word.lower()": token.lower(),
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
        "is_phone_like": bool(re.match(r"^(?:\+7|8)\d+$", token)),
        "is_email_like": "@" in token and "." in token,
        "is_city_marker": token.lower() in {"город", "г", "москва", "казань", "санкт-петербург", "новосибирск"},
        "is_address_marker": token.lower() in {
            "адрес", "улица", "ул", "ул.", "проспект", "пр", "пр.", "дом", "д", "д.",
            "квартира", "кв", "кв.", "невский", "ленина", "тверская", "баумана"
        },
        "is_name_marker": token.lower() in {"зовут", "имя", "клиент", "это"},
        "is_contact_marker": token.lower() in {"телефон", "почта", "email", "паспорт"},
    }

    if i > 0:
        prev_token = tokens[i - 1].text
        features.update({
            "-1:word.lower()": prev_token.lower(),
            "-1:word.istitle()": prev_token.istitle(),
            "-1:is_address_marker": prev_token.lower() in {
                "адрес", "улица", "ул", "ул.", "проспект", "дом", "д", "кв", "квартира"
            },
            "-1:is_name_marker": prev_token.lower() in {"зовут", "это", "клиент"},
        })
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        next_token = tokens[i + 1].text
        features.update({
            "+1:word.lower()": next_token.lower(),
            "+1:word.istitle()": next_token.istitle(),
            "+1:is_address_marker": next_token.lower() in {
                "улица", "ул", "ул.", "проспект", "дом", "д", "квартира", "кв"
            },
        })
    else:
        features["EOS"] = True

    return features


def sent2features(tokens: List[Token]) -> List[Dict[str, object]]:
    return [token2features(tokens, i) for i in range(len(tokens))]


def bio_to_entities(text: str, tokens: List[Token], labels: List[str]) -> List[EntityMatch]:
    entities = []
    current_type = None
    start_idx = None
    end_idx = None

    for i, label in enumerate(labels):
        if label == "O":
            if current_type is not None:
                entities.append(
                    EntityMatch(
                        entity_type=current_type,
                        start=tokens[start_idx].start,
                        end=tokens[end_idx].end,
                        text=text[tokens[start_idx].start:tokens[end_idx].end],
                        score=0.80,
                        source="crf_model",
                    )
                )
                current_type = None
                start_idx = None
                end_idx = None
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B":
            if current_type is not None:
                entities.append(
                    EntityMatch(
                        entity_type=current_type,
                        start=tokens[start_idx].start,
                        end=tokens[end_idx].end,
                        text=text[tokens[start_idx].start:tokens[end_idx].end],
                        score=0.80,
                        source="crf_model",
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

    if current_type is not None:
        entities.append(
            EntityMatch(
                entity_type=current_type,
                start=tokens[start_idx].start,
                end=tokens[end_idx].end,
                text=text[tokens[start_idx].start:tokens[end_idx].end],
                score=0.80,
                source="crf_model",
            )
        )

    return entities


def predict_entities(text: str, model_path: str = MODEL_PATH) -> List[EntityMatch]:
    crf = joblib.load(model_path)
    tokens = tokenize(text)
    X = [sent2features(tokens)]
    y_pred = crf.predict(X)[0]
    return bio_to_entities(text, tokens, y_pred)


if __name__ == "__main__":
    while True:
        text = input("Введите текст (/exit для выхода):\n> ").strip()
        if text == "/exit":
            break

        entities = predict_entities(text)
        print("\nНайдено сущностей:")
        if not entities:
            print("Ничего не найдено")
        else:
            for e in entities:
                print(f"{e.entity_type}: '{e.text}' [{e.start}, {e.end}]")
        print()