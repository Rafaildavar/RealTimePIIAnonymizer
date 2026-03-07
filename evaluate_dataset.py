import re
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable
from collections import defaultdict


# =========================================================
# 1. Структура сущности
# =========================================================

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


# =========================================================
# 2. Базовый recognizer
# =========================================================

class BaseRecognizer:
    name: str = "base"

    def find(self, text: str) -> List[EntityMatch]:
        raise NotImplementedError


# =========================================================
# 3. Regex recognizers
# =========================================================

class RegexRecognizer(BaseRecognizer):
    def __init__(self, name: str, entity_type: str, pattern: str, score: float):
        self.name = name
        self.entity_type = entity_type
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.score = score

    def find(self, text: str) -> List[EntityMatch]:
        matches = []
        for m in self.pattern.finditer(text):
            matches.append(
                EntityMatch(
                    entity_type=self.entity_type,
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                    score=self.score,
                    source=self.name,
                )
            )
        return matches


class ContextRegexRecognizer(BaseRecognizer):
    def __init__(self, name: str, entity_type: str, pattern: str, score: float, context_words: List[str]):
        self.name = name
        self.entity_type = entity_type
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.base_score = score
        self.context_words = [w.lower() for w in context_words]

    def find(self, text: str) -> List[EntityMatch]:
        text_lower = text.lower()
        matches = []

        for m in self.pattern.finditer(text):
            left = max(0, m.start() - 40)
            right = min(len(text), m.end() + 40)
            window = text_lower[left:right]

            boost = 0.08 if any(word in window for word in self.context_words) else 0.0

            matches.append(
                EntityMatch(
                    entity_type=self.entity_type,
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                    score=min(0.99, self.base_score + boost),
                    source=self.name,
                )
            )

        return matches


# =========================================================
# 4. Эвристика для ФИО
# =========================================================

class NameHeuristicRecognizer(BaseRecognizer):
    name = "name_heuristic"

    def __init__(self, first_names: Iterable[str], last_names: Iterable[str]):
        self.first_names = set(n.lower() for n in first_names)
        self.last_names = set(n.lower() for n in last_names)
        self.pattern = re.compile(r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}\b")

    def find(self, text: str) -> List[EntityMatch]:
        matches = []

        for m in self.pattern.finditer(text):
            candidate = m.group().split()
            lowered = [x.lower() for x in candidate]

            score = 0.55
            if len(lowered) >= 2:
                if lowered[0] in self.first_names:
                    score += 0.18
                if lowered[1] in self.last_names:
                    score += 0.18
                if any(x in self.first_names for x in lowered):
                    score += 0.05

            if score >= 0.70:
                matches.append(
                    EntityMatch(
                        entity_type="NAME",
                        start=m.start(),
                        end=m.end(),
                        text=m.group(),
                        score=min(score, 0.95),
                        source=self.name,
                    )
                )

        return matches


# =========================================================
# 5. Эвристика для адреса
# =========================================================

class AddressHeuristicRecognizer(BaseRecognizer):
    name = "address_heuristic"

    def __init__(self):
        self.pattern = re.compile(
            r"\b(?:г\.?|город|ул\.?|улица|проспект|пр\.?|пер\.?|переулок|д\.?|дом|кв\.?|квартира|адрес)\s+[^,.!\n]{3,100}",
            re.IGNORECASE,
        )

    def find(self, text: str) -> List[EntityMatch]:
        matches = []
        for m in self.pattern.finditer(text):
            matches.append(
                EntityMatch(
                    entity_type="ADDRESS",
                    start=m.start(),
                    end=m.end(),
                    text=m.group().strip(),
                    score=0.82,
                    source=self.name,
                )
            )
        return matches


# =========================================================
# 6. Resolver конфликтов
# =========================================================

ENTITY_PRIORITY = {
    "CARD": 100,
    "PASSPORT": 95,
    "PHONE": 90,
    "EMAIL": 88,
    "SNILS": 87,
    "INN": 86,
    "DOB": 80,
    "ACCOUNT": 75,
    "ADDRESS": 70,
    "NAME": 65,
}


class ConflictResolver:
    @staticmethod
    def overlaps(a: EntityMatch, b: EntityMatch) -> bool:
        return not (a.end <= b.start or b.end <= a.start)

    @staticmethod
    def better(a: EntityMatch, b: EntityMatch) -> EntityMatch:
        if a.score != b.score:
            return a if a.score > b.score else b
        if a.length != b.length:
            return a if a.length > b.length else b
        pa = ENTITY_PRIORITY.get(a.entity_type, 0)
        pb = ENTITY_PRIORITY.get(b.entity_type, 0)
        return a if pa >= pb else b

    def resolve(self, matches: List[EntityMatch]) -> List[EntityMatch]:
        matches = sorted(matches, key=lambda x: (x.start, -(x.end - x.start)))
        result = []

        for current in matches:
            replaced = False
            for i, existing in enumerate(result):
                if self.overlaps(current, existing):
                    result[i] = self.better(current, existing)
                    replaced = True
                    break
            if not replaced:
                result.append(current)

        result.sort(key=lambda x: x.start)
        return result


# =========================================================
# 7. Masker
# =========================================================

class PIIMasker:
    def __init__(self):
        self.counters = defaultdict(int)

    def reset_counters(self):
        self.counters = defaultdict(int)

    def mask(self, text: str, matches: List[EntityMatch]) -> Tuple[str, Dict[str, str], List[EntityMatch]]:
        matches = sorted(matches, key=lambda x: x.start)
        offset = 0
        out = text
        mapping = {}
        updated_matches = []

        for m in matches:
            self.counters[m.entity_type] += 1
            tag = f"[{m.entity_type}_{self.counters[m.entity_type]}]"

            start = m.start + offset
            end = m.end + offset

            out = out[:start] + tag + out[end:]
            mapping[tag] = m.text

            new_end = start + len(tag)
            offset += len(tag) - (m.end - m.start)

            updated_matches.append(
                EntityMatch(
                    entity_type=m.entity_type,
                    start=start,
                    end=new_end,
                    text=tag,
                    score=m.score,
                    source=m.source,
                )
            )

        return out, mapping, updated_matches


# =========================================================
# 8. Главный движок
# =========================================================

class PIIDetectorEngine:
    def __init__(self, recognizers: List[BaseRecognizer]):
        self.recognizers = recognizers
        self.resolver = ConflictResolver()

    def detect(self, text: str) -> List[EntityMatch]:
        all_matches = []
        for recognizer in self.recognizers:
            all_matches.extend(recognizer.find(text))
        return self.resolver.resolve(all_matches)


# =========================================================
# 9. Конфигурация
# =========================================================

def build_engine() -> PIIDetectorEngine:
    first_names = [
        "иван", "петр", "сергей", "анна", "мария", "елена",
        "дмитрий", "алексей", "андрей", "максим", "наталья",
        "екатерина", "виктория", "михаил"
    ]
    last_names = [
        "иванов", "петров", "сидоров", "кузнецова", "смирнов",
        "волков", "соколов", "попов", "федоров", "орлов",
        "сергеев", "алексеев"
    ]

    recognizers = [
        RegexRecognizer(
            name="email_regex",
            entity_type="EMAIL",
            pattern=r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            score=0.98,
        ),
        ContextRegexRecognizer(
            name="phone_regex",
            entity_type="PHONE",
            pattern=r"(?<!\d)(?:\+7|8)\d{10}(?!\d)|(?<!\d)(?:\+7|8)[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)",
            score=0.95,
            context_words=["телефон", "номер телефона", "номер", "контактный номер", "для связи"],
        ),
        ContextRegexRecognizer(
            name="passport_regex",
            entity_type="PASSPORT",
            pattern=r"(?<!\d)\d{4}\s\d{6}(?!\d)|(?<!\d)\d{10}(?!\d)",
            score=0.92,
            context_words=["паспорт", "паспортные данные", "паспортные", "личность"],
        ),
        RegexRecognizer(
            name="card_regex",
            entity_type="CARD",
            pattern=r"(?<!\d)(?:\d{4}\s\d{4}\s\d{4}\s\d{4})(?!\d)",
            score=0.96,
        ),
        NameHeuristicRecognizer(first_names=first_names, last_names=last_names),
        AddressHeuristicRecognizer(),
    ]

    return PIIDetectorEngine(recognizers)


# =========================================================
# 10. Нормализация тегов
# =========================================================

def normalize_masked_text(text: str) -> str:
    return re.sub(r"\[(\w+)_\d+\]", r"[\1]", text)


# =========================================================
# 11. Загрузка датасета
# =========================================================

def load_dataset(path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "/r/" not in line:
                continue
            text, target = line.split("/r/", 1)
            rows.append((text.strip(), target.strip()))
    return rows


# =========================================================
# 12. Сохранение результатов
# =========================================================

def save_results_to_csv(results: List[Dict], path: str) -> None:
    fieldnames = [
        "id",
        "text",
        "target",
        "pred",
        "is_match",
        "mapping",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def save_errors_to_csv(results: List[Dict], path: str) -> None:
    errors = [row for row in results if row["is_match"] == 0]
    save_results_to_csv(errors, path)


def save_metrics(total: int, exact: int, accuracy: float, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Всего примеров: {total}\n")
        f.write(f"Точных совпадений: {exact}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Ошибок: {total - exact}\n")


# =========================================================
# 13. Оценка датасета
# =========================================================

def evaluate_dataset(path: str):
    engine = build_engine()
    masker = PIIMasker()

    rows = load_dataset(path)

    total = len(rows)
    exact = 0
    results = []

    for idx, (text, target) in enumerate(rows, start=1):
        masker.reset_counters()

        matches = engine.detect(text)
        masked_text, mapping, _ = masker.mask(text, matches)
        pred = normalize_masked_text(masked_text)

        is_match = int(pred == target)
        if is_match:
            exact += 1

        results.append({
            "id": idx,
            "text": text,
            "target": target,
            "pred": pred,
            "is_match": is_match,
            "mapping": str(mapping),
        })

    accuracy = exact / total if total else 0.0

    print("=" * 80)
    print(f"Всего примеров: {total}")
    print(f"Точных совпадений: {exact}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Ошибок: {total - exact}")
    print("=" * 80)

    save_results_to_csv(results, "analysis_results.csv")
    save_errors_to_csv(results, "errors_only.csv")
    save_metrics(total, exact, accuracy, "metrics.txt")

    print("Файлы сохранены:")
    print(" - analysis_results.csv")
    print(" - errors_only.csv")
    print(" - metrics.txt")


if __name__ == "__main__":
    evaluate_dataset("data_pers.csv")