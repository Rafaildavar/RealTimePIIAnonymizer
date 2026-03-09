import re
import joblib
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pathlib import Path

import sklearn_crfsuite
from sklearn_crfsuite import metrics


# =========================================================
# 1. Настройки
# =========================================================

DATASET_PATH = "data_pers.csv"
MODEL_PATH = "crf_pii_model.joblib"


# =========================================================
# 2. Токенизация
# =========================================================

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> List[Token]:
    return [Token(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


# =========================================================
# 3. Загрузка датасета
# Формат строки:
# text/r/target
# =========================================================

def load_dataset(path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "/r/" not in line:
                continue

            text, target = line.split("/r/", 1)
            text = text.strip()
            target = target.strip()

            # пропускаем заголовок
            if text.lower() == "text" and target.lower() == "target":
                continue

            rows.append((text, target))
    return rows


# =========================================================
# 4. Выравнивание target -> spans в original text
#
# Пример:
# text   = "Здравствуйте меня зовут Иван Иванов ..."
# target = "Здравствуйте меня зовут [NAME] ..."
#
# На выходе:
# [("NAME", start, end), ...]
# =========================================================

PLACEHOLDER_PATTERN = re.compile(r"\[(\w+)\]")


def extract_entity_spans_from_pair(text: str, target: str) -> List[Tuple[str, int, int]]:
    spans = []

    i = 0  # pointer in text
    j = 0  # pointer in target

    while j < len(target):
        ph = PLACEHOLDER_PATTERN.match(target, j)
        if ph:
            entity_type = ph.group(1)

            next_target_pos = ph.end()

            # Найдём следующий "обычный" кусок target после placeholder
            next_literal_start = next_target_pos
            next_ph = PLACEHOLDER_PATTERN.search(target, next_literal_start)

            if next_ph:
                next_literal = target[next_literal_start:next_ph.start()]
            else:
                next_literal = target[next_literal_start:]

            next_literal = next_literal

            if next_literal:
                next_idx_in_text = text.find(next_literal, i)
                if next_idx_in_text == -1:
                    raise ValueError(
                        f"Не удалось выровнять placeholder [{entity_type}] "
                        f"в тексте.\nTEXT: {text}\nTARGET: {target}"
                    )
                spans.append((entity_type, i, next_idx_in_text))
                i = next_idx_in_text
            else:
                # Placeholder в конце строки: сущность до конца текста
                spans.append((entity_type, i, len(text)))
                i = len(text)

            j = ph.end()
        else:
            if i >= len(text):
                break

            if text[i] == target[j]:
                i += 1
                j += 1
            else:
                # мягкое выравнивание
                next_match = text.find(target[j], i)
                if next_match == -1:
                    raise ValueError(
                        f"Не удалось выровнять текст.\nTEXT: {text}\nTARGET: {target}\n"
                        f"text_pos={i}, target_pos={j}"
                    )
                i = next_match

    return spans


# =========================================================
# 5. Преобразование spans -> BIO labels по токенам
# =========================================================

def spans_to_bio_labels(tokens: List[Token], spans: List[Tuple[str, int, int]]) -> List[str]:
    labels = ["O"] * len(tokens)

    for entity_type, ent_start, ent_end in spans:
        token_indices = []
        for idx, token in enumerate(tokens):
            if not (token.end <= ent_start or token.start >= ent_end):
                token_indices.append(idx)

        if not token_indices:
            continue

        labels[token_indices[0]] = f"B-{entity_type}"
        for idx in token_indices[1:]:
            labels[idx] = f"I-{entity_type}"

    return labels


# =========================================================
# 6. Признаки для CRF
# =========================================================

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


# =========================================================
# 7. Подготовка выборки
# =========================================================

def build_crf_dataset(rows: List[Tuple[str, str]]):
    X = []
    y = []

    for text, target in rows:
        tokens = tokenize(text)
        spans = extract_entity_spans_from_pair(text, target)
        labels = spans_to_bio_labels(tokens, spans)

        X.append(sent2features(tokens))
        y.append(labels)

    return X, y


# =========================================================
# 8. Обучение
# =========================================================

def train():
    rows = load_dataset(DATASET_PATH)
    print(f"Загружено примеров: {len(rows)}")

    X, y = build_crf_dataset(rows)

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=200,
        all_possible_transitions=True,
    )

    crf.fit(X, y)

    y_pred = crf.predict(X)

    labels = list(crf.classes_)
    labels = [label for label in labels if label != "O"]

    print("\n=== Classification report on train ===")
    print(metrics.flat_classification_report(y, y_pred, labels=labels, digits=4))

    joblib.dump(crf, MODEL_PATH)
    print(f"\nМодель сохранена в: {MODEL_PATH}")


if __name__ == "__main__":
    train()