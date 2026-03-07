import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Iterable
from collections import defaultdict


# =========================
# 1. Структура сущности
# =========================

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


# =========================
# 2. Базовый recognizer
# =========================

class BaseRecognizer:
    name: str = "base"

    def find(self, text: str) -> List[EntityMatch]:
        raise NotImplementedError


# =========================
# 3. Regex recognizers
# =========================

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


# =========================
# 4. Эвристика для ФИО
# =========================

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


# =========================
# 5. Эвристика для адреса
# =========================

class AddressHeuristicRecognizer(BaseRecognizer):
    name = "address_heuristic"

    def __init__(self):
        self.pattern = re.compile(
            r"\b(?:г\.?|город|ул\.?|улица|проспект|пр\.?|пер\.?|переулок|д\.?|дом|кв\.?|квартира)\s+[^,.\n]{3,80}",
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


# =========================
# 6. Resolver конфликтов
# =========================

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


# =========================
# 7. Masker / Restore
# =========================

class PIIMasker:
    def __init__(self):
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

    @staticmethod
    def restore(text: str, mapping: Dict[str, str]) -> str:
        for tag, original in mapping.items():
            text = text.replace(tag, original)
        return text


# =========================
# 8. Streaming restore
# =========================

class StreamingRestoreEngine:
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping
        self.buffer = ""

    def feed(self, chunk: str) -> str:
        self.buffer += chunk

        safe_upto = self._safe_cut_position(self.buffer)
        ready = self.buffer[:safe_upto]
        self.buffer = self.buffer[safe_upto:]

        for tag, original in self.mapping.items():
            ready = ready.replace(tag, original)

        return ready

    def flush(self) -> str:
        tail = self.buffer
        self.buffer = ""
        for tag, original in self.mapping.items():
            tail = tail.replace(tag, original)
        return tail

    @staticmethod
    def _safe_cut_position(text: str) -> int:
        last_open = text.rfind("[")
        last_close = text.rfind("]")
        if last_open > last_close:
            return last_open
        return len(text)


# =========================
# 9. Главный движок
# =========================

class PIIDetectorEngine:
    def __init__(self, recognizers: List[BaseRecognizer]):
        self.recognizers = recognizers
        self.resolver = ConflictResolver()

    def detect(self, text: str) -> List[EntityMatch]:
        all_matches = []
        for recognizer in self.recognizers:
            all_matches.extend(recognizer.find(text))
        return self.resolver.resolve(all_matches)


# =========================
# 10. Конфигурация
# =========================

def build_engine() -> PIIDetectorEngine:
    first_names = [
        "иван", "анна", "мария", "петр", "сергей", "елена",
        "дмитрий", "олег", "наталья", "алексей"
    ]
    last_names = [
        "иванов", "петров", "сидоров", "смирнов",
        "кузнецов", "попов", "волков", "соколов"
    ]

    recognizers = [
        ContextRegexRecognizer(
            name="passport_regex",
            entity_type="PASSPORT",
            pattern=r"\b\d{4}\s?\d{6}\b",
            score=0.92,
            context_words=["паспорт", "серия", "номер паспорта"],
        ),
        ContextRegexRecognizer(
            name="phone_regex",
            entity_type="PHONE",
            pattern=r"(?:\+7|8)[\s\-(]*\d{3}[\s\)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}",
            score=0.95,
            context_words=["телефон", "номер", "связаться"],
        ),
        RegexRecognizer(
            name="email_regex",
            entity_type="EMAIL",
            pattern=r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            score=0.98,
        ),
        ContextRegexRecognizer(
            name="dob_regex",
            entity_type="DOB",
            pattern=r"\b(?:0?[1-9]|[12][0-9]|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:19\d{2}|20\d{2})\b",
            score=0.88,
            context_words=["дата рождения", "родился", "родилась", "др"],
        ),
        RegexRecognizer(
            name="card_regex",
            entity_type="CARD",
            pattern=r"\b(?:\d{4}[ -]?){3}\d{4}\b",
            score=0.96,
        ),
        ContextRegexRecognizer(
            name="account_regex",
            entity_type="ACCOUNT",
            pattern=r"\b\d{10,20}\b",
            score=0.78,
            context_words=["счет", "договор", "лицевой счет", "номер договора"],
        ),
        NameHeuristicRecognizer(first_names=first_names, last_names=last_names),
        AddressHeuristicRecognizer(),
    ]

    return PIIDetectorEngine(recognizers)


# =========================
# 11. Простая оценка
# =========================

def evaluate_span_level(pred: List[EntityMatch], gold: List[EntityMatch]) -> Dict[str, float]:
    pred_set = {(x.start, x.end, x.entity_type) for x in pred}
    gold_set = {(x.start, x.end, x.entity_type) for x in gold}

    correct_full = pred_set & gold_set

    pred_span = {(x.start, x.end) for x in pred}
    gold_span = {(x.start, x.end) for x in gold}
    correct_span = pred_span & gold_span

    recall = len(correct_span) / len(gold_span) if gold_span else 1.0
    type_precision = len(correct_full) / len(pred_set) if pred_set else 1.0
    score = (recall + type_precision) / 2

    return {
        "recall": round(recall, 4),
        "type_precision": round(type_precision, 4),
        "final_score": round(score, 4),
    }


# =========================
# 12. Демонстрация
# =========================

def main():
    text = (
        "Меня зовут Иван Иванов, дата рождения 12.03.2001, "
        "мой паспорт 1234 567890, телефон +7 999 123-45-67, "
        "email ivan@example.com, "
        "живу по адресу г. Санкт-Петербург, ул. Ленина, д. 10. "
        "Номер счета 40817810099910004312."
    )

    engine = build_engine()
    matches = engine.detect(text)

    print("=" * 80)
    print("ИСХОДНЫЙ ТЕКСТ")
    print(text)
    print("=" * 80)

    print("\nНАЙДЕННЫЕ СУЩНОСТИ:")
    for m in matches:
        print(asdict(m))

    masker = PIIMasker()
    masked_text, mapping, masked_matches = masker.mask(text, matches)

    print("\n" + "=" * 80)
    print("MASKED TEXT")
    print(masked_text)
    print("=" * 80)

    print("\nMAPPING:")
    for k, v in mapping.items():
        print(f"{k} -> {v}")

    llm_response = (
        "Здравствуйте, [NAME_1]! "
        "Мы видим, что дата рождения [DOB_1] и паспорт [PASSPORT_1] были указаны. "
        "Для связи используем [PHONE_1]."
    )

    restored = masker.restore(llm_response, mapping)

    print("\n" + "=" * 80)
    print("LLM RESPONSE")
    print(llm_response)
    print("=" * 80)

    print("\nRESTORED RESPONSE:")
    print(restored)

    print("\n" + "=" * 80)
    print("STREAMING DEMO")
    print("=" * 80)

    stream = StreamingRestoreEngine(mapping)
    chunks = [
        "Здравствуйте, [NA",
        "ME_1]! Ваш запрос об",
        "работан. Паспорт [PASSPORT_1] проверен."
    ]

    for i, c in enumerate(chunks, start=1):
        out = stream.feed(c)
        print(f"chunk {i}: {repr(c)}")
        if out:
            print("OUT:", out)

    tail = stream.flush()
    if tail:
        print("OUT:", tail)


if __name__ == "__main__":
    main()