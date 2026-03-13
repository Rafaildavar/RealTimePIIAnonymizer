import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from unmask import unmask_pii
from metrics import metrics


# Небольшой стартовый словарь имен.
# Лучше вынести в отдельный JSON/TXT и расширять.
FIRST_NAMES = {
    "александр", "алексей", "алёна", "алена", "анастасия", "андрей", "анна",
    "антон", "артём", "артем", "вадим", "валерия", "виктор", "виктория",
    "виталий", "владимир", "владислав", "даниил", "дарья", "дмитрий",
    "евгений", "екатерина", "елена", "иван", "игорь", "илья", "кирилл",
    "ксения", "лариса", "максим", "мария", "михаил", "наталья", "никита",
    "николай", "оксана", "ольга", "павел", "полина", "роман", "светлана",
    "сергей", "софия", "татьяна", "тимур", "фёдор", "федор", "юлия", "яна",
    "денис", "надежда", "марина",
}

# Слова-маркеры, после которых часто идет ФИО.
NAME_MARKERS = (
    "меня зовут",
    "мое имя",
    "моё имя",
    "фио",
    "ф.и.о.",
    "получатель",
    "клиент",
    "заказчик",
    "пациент",
    "сотрудник",
    "контактное лицо",
    "моя фамилия",
    "фамилия",
)

# Слишком общие слова, которые не стоит считать ФИО.
STOPWORDS = {
    "город", "улица", "дом", "квартира", "область", "район", "республика",
    "почта", "email", "телефон", "номер", "карта", "паспорт", "адрес",
    "клиент", "получатель", "заказчик", "пациент", "сотрудник",
}

# Для токенизации слов и инициалов вида "И." / "И.И."
TOKEN_RE = re.compile(
    r"[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)?|[А-ЯЁ]\.[А-ЯЁ]\.|[А-ЯЁ]\.",
    re.UNICODE,
)

INITIALS_RE = re.compile(r"^[А-ЯЁ]\.(?:[А-ЯЁ]\.)?$")
CYRILLIC_WORD_RE = re.compile(r"^[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)?$")


@dataclass(frozen=True)
class MaskResult:
    """
    Результат маскирования: текст с плейсхолдерами и словарь для обратной подстановки.
    mapping хранится в формате {"[NAME_n]": "исходное значение"}.
    """
    masked_text: str
    mapping: Dict[str, str]


def _make_placeholder(entity: str, counters: Dict[str, int]) -> str:
    """
    Генерирует уникальный плейсхолдер для сущности и увеличивает счетчик.
    Пример: для entity="NAME" вернет "[NAME_1]", затем "[NAME_2]".
    """
    counters[entity] = counters.get(entity, 0) + 1
    return f"[{entity}_{counters[entity]}]"


def _is_initials(token: str) -> bool:
    """Проверяет, является ли токен инициалами: 'И.' или 'И.И.'."""
    return bool(INITIALS_RE.fullmatch(token))


def _is_cyrillic_word(token: str) -> bool:
    """Проверяет, что токен состоит из кириллических букв и, возможно, дефиса."""
    return bool(CYRILLIC_WORD_RE.fullmatch(token))


def _is_capitalized_word(token: str) -> bool:
    """
    Проверяет, что слово похоже на имя/фамилию с заглавной буквы:
    'Анна', 'Петрова', 'Анна-Мария'.
    """
    if not _is_cyrillic_word(token):
        return False

    parts = token.split("-")
    return all(part[:1].isupper() and part[1:].islower() for part in parts if part)


def _looks_like_patronymic(token: str) -> bool:
    """Грубая эвристика для отчеств."""
    t = token.lower()
    return _is_cyrillic_word(token) and t.endswith(("вич", "вна", "ична", "оглы", "кызы"))


def _looks_like_surname(token: str) -> bool:
    """Грубая эвристика для русских фамилий."""
    t = token.lower()
    surname_suffixes = (
        "ов", "ова", "ев", "ева", "ин", "ина", "ын", "ына",
        "ский", "ская", "цкий", "цкая", "енко", "ко", "ук", "юк",
        "ян", "дзе", "иди", "швили", "ову",
    )
    return _is_cyrillic_word(token) and t.endswith(surname_suffixes)


def _is_first_name(token: str) -> bool:
    """Проверяет, есть ли токен в словаре имен."""
    return token.lower() in FIRST_NAMES


def _tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Возвращает список токенов с их позициями в тексте."""
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _has_name_marker(left_context: str) -> bool:
    """Проверяет, есть ли перед кандидатом контекстный маркер ФИО."""
    lowered = left_context.lower()
    return any(marker in lowered for marker in NAME_MARKERS)


def _score_name_tokens(tokens: List[str], has_context: bool) -> int:
    """
    Возвращает score кандидата на ФИО.
    Чем выше score, тем больше уверенность, что перед нами имя/ФИО.
    """
    if not tokens or len(tokens) > 3:
        return 0

    lowered = [t.lower() for t in tokens]

    # Стоп-слова отсекаем сразу.
    if any(t in STOPWORDS for t in lowered):
        return 0

    initials_count = sum(_is_initials(t) for t in tokens)
    capitalized_count = sum(_is_capitalized_word(t) for t in tokens)
    first_name_count = sum(_is_first_name(t) for t in tokens)
    patronymic_count = sum(_looks_like_patronymic(t) for t in tokens)
    surname_count = sum(_looks_like_surname(t) for t in tokens)

    # Для кандидатов без контекста требуем "приличный" вид:
    # либо слова с заглавной буквы, либо инициалы.
    if not has_context:
        if capitalized_count + initials_count != len(tokens):
            return 0

    score = 0
    score += first_name_count * 3
    score += patronymic_count * 3
    score += surname_count * 2
    score += initials_count * 2
    score += capitalized_count
    if has_context:
        score += 3

    if len(tokens) == 3:
        # Хороший классический случай: Имя Отчество Фамилия / Фамилия Имя Отчество
        if patronymic_count >= 1 and (first_name_count >= 1 or surname_count >= 1):
            return score

        # Еще один допустимый случай: имя + фамилия + инициалы / и т.п.
        if (first_name_count + surname_count + initials_count) >= 3:
            return score

        return 0

    if len(tokens) == 2:
        # Имя Фамилия / Фамилия Имя
        if first_name_count >= 1 and surname_count >= 1:
            return score

        # Фамилия И.О. / И.О. Фамилия
        if surname_count >= 1 and initials_count >= 1:
            return score

        # После явного маркера допускаем "Имя Отчество"
        if has_context and first_name_count >= 1 and patronymic_count >= 1:
            return score

        # После маркера можно чуть ослабить условия
        if has_context and (first_name_count + surname_count + patronymic_count) >= 2:
            return score

        return 0

    if len(tokens) == 1:
        # Один токен разрешаем только после сильного контекста:
        # "меня зовут Алексей", "сменила фамилию на Иванову"
        if has_context and (first_name_count >= 1 or surname_count >= 1 or patronymic_count >= 1):
            return score
        return 0

    return 0


def _collect_context_candidates(text: str) -> List[Tuple[int, int, int]]:
    """
    Ищет ФИО после контекстных маркеров:
    'меня зовут ...', 'фио: ...', 'получатель: ...' и т.п.
    """
    marker_re = re.compile(
        r"(?i)\b(?:"
        r"меня\s+зовут|мо[её]\s+имя|фио|ф\.и\.о\.|получатель|клиент|заказчик|пациент|сотрудник|контактное\s+лицо|"
        r"сменил\s+фамилию\s+на|сменила\s+фамилию\s+на|моя\s+фамилия|фамилия|я"
        r")\b\s*[:\-]?\s*"
    )

    candidates: List[Tuple[int, int, int]] = []

    for match in marker_re.finditer(text):
        start = match.end()

        # Смотрим небольшой хвост после маркера и пробуем взять 2-3 токена.
        tail = text[start:start + 80]
        tokens = _tokenize_with_spans(tail)

        for window_size in (3, 2, 1):
            if len(tokens) < window_size:
                continue

            window = tokens[:window_size]
            candidate_tokens = [tok for tok, _, _ in window]

            # Берем только если между токенами нет "тяжелой" пунктуации.
            ok = True
            for idx in range(1, len(window)):
                prev_end = window[idx - 1][2]
                curr_start = window[idx][1]
                between = tail[prev_end:curr_start]
                if between.strip() and not between.isspace():
                    ok = False
                    break

            if not ok:
                continue

            score = _score_name_tokens(candidate_tokens, has_context=True)
            if score <= 0:
                continue

            abs_start = start + window[0][1]
            abs_end = start + window[-1][2]
            candidates.append((abs_start, abs_end, score))
            break

    return candidates


def _collect_generic_candidates(text: str) -> List[Tuple[int, int, int]]:
    """
    Ищет ФИО в свободном тексте без явного маркера.
    Использует окна из 2-3 токенов и набор эвристик.
    """
    tokens = _tokenize_with_spans(text)
    candidates: List[Tuple[int, int, int]] = []

    for i in range(len(tokens)):
        for window_size in (3, 2):
            if i + window_size > len(tokens):
                continue

            window = tokens[i:i + window_size]
            candidate_tokens = [tok for tok, _, _ in window]

            # Между токенами допускаем только пробелы/переводы строк.
            contiguous = True
            for idx in range(1, len(window)):
                prev_end = window[idx - 1][2]
                curr_start = window[idx][1]
                between = text[prev_end:curr_start]
                if between.strip():
                    contiguous = False
                    break

            if not contiguous:
                continue

            # Небольшой левый контекст: вдруг перед окном есть маркер
            left_context = text[max(0, window[0][1] - 40):window[0][1]]
            has_context = _has_name_marker(left_context)

            score = _score_name_tokens(candidate_tokens, has_context=has_context)
            if score <= 0:
                continue

            candidates.append((window[0][1], window[-1][2], score))

    return candidates


def _select_non_overlapping(candidates: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """
    Выбирает непересекающиеся кандидаты.
    Приоритет у более длинных и более уверенных совпадений.
    """
    def _sort_key(item: Tuple[int, int, int]) -> Tuple[int, int, int]:
        start, end, score = item
        length = end - start
        return (-length, -score, start)

    selected: List[Tuple[int, int, int]] = []
    occupied: List[Tuple[int, int]] = []

    for candidate in sorted(candidates, key=_sort_key):
        start, end, _ = candidate

        overlaps = any(not (end <= occ_start or start >= occ_end) for occ_start, occ_end in occupied)
        if overlaps:
            continue

        selected.append(candidate)
        occupied.append((start, end))

    return sorted(selected, key=lambda x: x[0])


def _mask_fio(text: str, counters: Dict[str, int], mapping: Dict[str, str]) -> str:
    """Ищет ФИО и заменяет их на плейсхолдеры [NAME_n]."""
    context_candidates = _collect_context_candidates(text)
    generic_candidates = _collect_generic_candidates(text)

    candidates = _select_non_overlapping(context_candidates + generic_candidates)
    if not candidates:
        return text

    prepared: List[Tuple[int, int, str]] = []

    # СНАЧАЛА назначаем placeholder слева направо
    for start, end, _score in candidates:
        original = text[start:end]

        if len(original.strip()) < 3:
            continue

        placeholder = _make_placeholder("NAME", counters)
        mapping[placeholder] = original
        prepared.append((start, end, placeholder))

    masked = text

    # ПОТОМ заменяем справа налево, чтобы не съезжали индексы
    for start, end, placeholder in reversed(prepared):
        masked = masked[:start] + placeholder + masked[end:]

    return masked


def mask_fio_with_mapping(text: str) -> MaskResult:
    """Возвращает текст с замаскированными ФИО и mapping для обратной подстановки."""
    counters: Dict[str, int] = {}
    mapping: Dict[str, str] = {}

    masked = _mask_fio(text, counters=counters, mapping=mapping)
    return MaskResult(masked_text=masked, mapping=mapping)


def mask_fio(text: str) -> str:
    """Совместимая обертка: возвращает только замаскированный текст."""
    return mask_fio_with_mapping(text).masked_text


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

def normalize_placeholders(text: str) -> str:
    """
    Нормализует [NAME_1], [NAME_2], ... в [NAME].
    Это полезно, если в gold-разметке маски без номеров.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\[NAME_\d+\]", "[NAME]", text)


if __name__ == "__main__":
    text = (
        "Меня зовут Анна Ивановна Петрова. "
        "Получатель: Сергей Николаевич Иванов. "
        "Вчера я говорил с Мария Орлова. "
        "Еще один клиент - Дмитрий Алексеевич Смирнов."
    )

    result = mask_fio_with_mapping(text)
    print("masked:", result.masked_text)
    print("mapping:", result.mapping)
    print("restored:", unmask_pii(result.masked_text, result.mapping))

    df = pd.read_csv("data/simple_data_100.csv")

    # Приводим колонки к безопасному строковому виду
    df["original"] = df["original"].fillna("").astype(str)
    df["masked"] = df["masked"].fillna("").astype(str)

    df["pred"] = ""
    df["mapping"] = ""

    for idx, row in df.iterrows():
        res = mask_fio_with_mapping(row["original"])
        df.at[idx, "pred"] = res.masked_text
        df.at[idx, "mapping"] = json.dumps(res.mapping, ensure_ascii=False)

    # Нормализуем gold и pred для случая, если в эталоне плейсхолдеры без индексов
    df["gold_for_metrics"] = df["masked"].apply(normalize_placeholders)
    df["pred_for_metrics"] = df["pred"].apply(normalize_placeholders)

    print(f"ЭТА ЭВРИСТИКА ИЩЕТ И ЗАМЕНЯЕТ ТОЛЬКО ФИО")
    print(metrics(df["gold_for_metrics"], df["pred_for_metrics"]))

    # Берем только строки, где в эталоне есть [NAME],
    # но в предсказании [NAME] нет
    name_missed = df[
        df["gold_for_metrics"].str.contains(r"\[NAME\]", regex=True, na=False)
        & ~df["pred_for_metrics"].str.contains(r"\[NAME\]", regex=True, na=False)
        ].copy()

    print(f"\nКоличество строк, где [NAME] есть в эталоне, но не найден в pred: {len(name_missed)}")

    if not name_missed.empty:
        for idx, row in name_missed.iterrows():
            print("\n" + "=" * 80)
            print(f"INDEX: {idx}")
            print("ORIGINAL:")
            print(row["original"])
            print("\nGOLD:")
            print(row["masked"])
            print("\nPRED:")
            print(row["pred_for_metrics"])

    # Берем только строки, где в эталоне нет [NAME],
    # но в предсказании [NAME] есть
    name_false_positive = df[
        ~df["gold_for_metrics"].str.contains(r"\[NAME\]", regex=True, na=False)
        & df["pred_for_metrics"].str.contains(r"\[NAME\]", regex=True, na=False)
        ].copy()

    print(f"\nКоличество строк, где [NAME] нет в эталоне, но найден в pred: {len(name_false_positive)}")

    if not name_false_positive.empty:
        for idx, row in name_false_positive.iterrows():
            print("\n" + "=" * 80)
            print(f"INDEX: {idx}")
            print("ORIGINAL:")
            print(row["original"])
            print("\nGOLD:")
            print(row["masked"])
            print("\nPRED:")
            print(row["pred_for_metrics"])