import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable
from collections import defaultdict
from crf_recognizer import CRFRecognizer

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
# 7. Маскирование и восстановление
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

    @staticmethod
    def restore(text: str, mapping: Dict[str, str]) -> str:
        for tag, original in mapping.items():
            text = text.replace(tag, original)
        return text


# =========================================================
# 8. Streaming restore engine
# =========================================================

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


# =========================================================
# 9. Главный движок
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
# 10. Конфигурация
# =========================================================

def build_engine() -> PIIDetectorEngine:
    first_names = [
        "иван", "анна", "мария", "петр", "сергей", "елена",
        "дмитрий", "олег", "наталья", "алексей", "максим",
        "артем", "екатерина", "андрей", "михаил", "виктория"
    ]
    last_names = [
        "иванов", "петров", "сидоров", "смирнов", "кузнецов",
        "попов", "волков", "соколов", "орлов", "федоров",
        "алексеев", "сергеев"
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
            pattern=r"\b(?:\+7|8)\s*\(?\d{3}\)?[\s-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}\b",
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
            score=0.90,
            context_words=["счет", "договор", "лицевой счет", "номер договора"],
        ),
        NameHeuristicRecognizer(first_names=first_names, last_names=last_names),
        AddressHeuristicRecognizer(),

        # ML-слой
        CRFRecognizer("crf_pii_model.joblib", allowed_types={"NAME", "ADDRESS"}),
    ]

    return PIIDetectorEngine(recognizers)


# =========================================================
# 11. Вывод
# =========================================================

def print_entities(matches: List[EntityMatch]) -> None:
    if not matches:
        print("Найденных персональных данных нет.")
        return

    print("Найденные сущности:")
    for i, m in enumerate(matches, start=1):
        print(
            f"{i}. type={m.entity_type:<10} "
            f"text='{m.text}' "
            f"span=({m.start},{m.end}) "
            f"score={m.score:.2f} "
            f"source={m.source}"
        )


def print_mapping(mapping: Dict[str, str]) -> None:
    if not mapping:
        print("Таблица замен пуста.")
        return

    print("Таблица замен:")
    for tag, original in mapping.items():
        print(f"  {tag} -> {original}")


# =========================================================
# 12. Потоковый ввод ответа LLM
# =========================================================

def interactive_stream_restore(mapping: Dict[str, str]) -> None:
    stream = StreamingRestoreEngine(mapping)

    print("\nРежим потокового ответа LLM.")
    print("Вводите ответ кусками. После каждого куска программа сразу покажет восстановленный фрагмент.")
    print("Команды:")
    print("  /done   - завершить поток")
    print("  /cancel - отменить режим")

    while True:
        chunk = input("chunk > ")
        if chunk == "/cancel":
            print("Потоковый режим отменен.")
            return

        if chunk == "/done":
            tail = stream.flush()
            if tail:
                print("RESTORED:", tail)
            print("Потоковый режим завершен.")
            return

        restored_part = stream.feed(chunk)
        if restored_part:
            print("RESTORED:", restored_part)
        else:
            print("RESTORED: <ожидание завершения placeholder>")


# =========================================================
# 13. CLI
# =========================================================

def realtime_cli():
    engine = build_engine()
    masker = PIIMasker()

    print("=" * 80)
    print("REAL-TIME PII MASKING CLI")
    print("Введите запрос клиента. Программа сразу замаскирует PII перед отправкой в LLM.")
    print("Команды:")
    print("  /exit - выйти")
    print("=" * 80)

    while True:
        print("\nВведите запрос клиента:")
        user_text = input("> ").strip()

        if not user_text:
            print("Пустая строка. Введите текст.")
            continue

        if user_text.lower() == "/exit":
            print("Выход.")
            break

        masker.reset_counters()

        matches = engine.detect(user_text)
        masked_text, mapping, _ = masker.mask(user_text, matches)

        print("\n" + "-" * 80)
        print_entities(matches)
        print("-" * 80)
        print("Masked prompt для отправки в LLM:")
        print(masked_text)
        print("-" * 80)
        print_mapping(mapping)
        print("-" * 80)

        print("Выберите режим обработки ответа LLM:")
        print("  1 - вставить полный ответ целиком")
        print("  2 - вводить ответ потоково, кусками")
        print("  Enter - пропустить")

        mode = input("mode > ").strip()

        if mode == "1":
            llm_response = input("LLM response > ").rstrip()
            if llm_response:
                restored = masker.restore(llm_response, mapping)
                print("\nФинальный ответ клиенту:")
                print(restored)
            else:
                print("Ответ LLM пустой.")
        elif mode == "2":
            interactive_stream_restore(mapping)
        else:
            print("Ответ LLM пропущен. Ожидается следующий запрос.")


# =========================================================
# 14. Точка входа
# =========================================================

if __name__ == "__main__":
    realtime_cli()