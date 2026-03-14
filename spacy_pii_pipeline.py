"""
Unified spaCy Pipeline for Russian PII Detection
=================================================
Combines a custom regex matcher (EntityRuler-style) with a trained spaCy NER
in a single pipeline for detecting 30 categories of PII in Russian banking text.

Architecture:
    text → tok2vec → regex_pii_matcher → ner → entity_merger → output

Usage:
    python spacy_pii_pipeline.py --train           # Train the pipeline
    python spacy_pii_pipeline.py --evaluate         # Evaluate on dev set
    python spacy_pii_pipeline.py --predict           # Predict on test set
"""

import re
import csv
import ast
import json
import random
import argparse
import logging
import warnings
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import minibatch, compounding

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
#  LABEL MAPPING  (short spaCy label ↔ full competition label)
# =====================================================================

LABEL_TO_FULL = {
    "API_KEY":      "API ключи",
    "CVV":          "CVV/CVC",
    "EMAIL":        "Email",
    "DRIVER_LIC":   "Водительское удостоверение",
    "TEMP_ID":      "Временное удостоверение личности",
    "CITIZENSHIP":  "Гражданство и названия стран",
    "VEHICLE":      "Данные об автомобиле клиента",
    "ORG_DATA":     "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)",
    "CARD_EXP":     "Дата окончания срока действия карты",
    "REG_DATE":     "Дата регистрации по месту жительства или пребывания",
    "DOB":          "Дата рождения",
    "CARD_HOLDER":  "Имя держателя карты",
    "CODE_WORD":    "Кодовые слова",
    "BIRTHPLACE":   "Место рождения",
    "BANK_NAME":    "Наименование банка",
    "BANK_ACCT":    "Номер банковского счета",
    "CARD_NUM":     "Номер карты",
    "PHONE":        "Номер телефона",
    "OTP":          "Одноразовые коды",
    "PIN":          "ПИН код",
    "PASSWORD":     "Пароли",
    "PASSPORT":     "Паспортные данные",
    "ADDRESS":      "Полный адрес",
    "WORK_PERMIT":  "Разрешение на работу / визу",
    "SNILS":        "СНИЛС клиента",
    "INN":          "Сведения об ИНН",
    "BIRTH_CERT":   "Свидетельство о рождении",
    "RESIDENCE":    "Серия и номер вида на жительство",
    "MAG_STRIPE":   "Содержимое магнитной полосы",
    "FIO":          "ФИО",
}

FULL_TO_LABEL = {v: k for k, v in LABEL_TO_FULL.items()}
ALL_LABELS = sorted(LABEL_TO_FULL.keys())

# Categories best handled by regex (structured patterns)
REGEX_LABELS: Set[str] = {
    "EMAIL", "PHONE", "CARD_NUM", "BANK_ACCT", "SNILS", "INN",
    "CARD_EXP", "MAG_STRIPE", "API_KEY", "CVV", "PIN", "OTP",
    "DRIVER_LIC", "TEMP_ID", "BIRTH_CERT", "RESIDENCE", "WORK_PERMIT",
    "PASSPORT", "PASSWORD", "CODE_WORD", "ORG_DATA", "VEHICLE",
    "DOB", "REG_DATE",
}

# Categories that rely primarily on NER (contextual)
NER_LABELS: Set[str] = {
    "FIO", "ADDRESS", "BIRTHPLACE", "CITIZENSHIP", "BANK_NAME",
    "CARD_HOLDER",
}


# =====================================================================
#  DATA LOADING
# =====================================================================

def load_train_data(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["text"]
            target_str = row["target"].strip()
            if target_str == "[]":
                entities = []
            else:
                raw = ast.literal_eval(target_str)
                entities = []
                for start, end, full_label in raw:
                    label = FULL_TO_LABEL.get(full_label)
                    if label:
                        entities.append((start, end, label))
            entities = _clean_overlapping(entities)
            data.append({"text": text, "entities": entities})
    return data


def load_test_data(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({"id": int(row["id"]), "text": row["text"]})
    return data


def _clean_overlapping(entities: List[Tuple]) -> List[Tuple]:
    if len(entities) <= 1:
        return entities
    sorted_ents = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    cleaned = [sorted_ents[0]]
    for ent in sorted_ents[1:]:
        prev = cleaned[-1]
        if ent[0] >= prev[1]:
            cleaned.append(ent)
        elif (ent[1] - ent[0]) > (prev[1] - prev[0]):
            cleaned[-1] = ent
    return cleaned


def split_data(data: List[Dict], dev_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    n = int(len(shuffled) * (1 - dev_ratio))
    return shuffled[:n], shuffled[n:]


# =====================================================================
#  REGEX PATTERNS  (character-level, context-aware)
# =====================================================================
# Each pattern is (compiled_regex, context_regex_or_None, search_radius)
# context_regex is checked against the full text; if None, pattern applies unconditionally.

_MONTHS_RU = (
    r"(?:январ[яьие]|феврал[яьие]|март[аеу]?|апрел[яьие]|"
    r"ма[яйюе]|июн[яьие]|июл[яьие]|август[аеу]?|"
    r"сентябр[яьие]|октябр[яьие]|ноябр[яьие]|декабр[яьие])"
)

_DATE_DMY = r"\d{2}\.\d{2}\.\d{4}"
_DATE_TEXT = r"\d{1,2}\s" + _MONTHS_RU + r"\s\d{4}(?:\s?года?)?"


def _build_regex_rules() -> Dict[str, list]:
    """Return {label: [(compiled_pattern, context_pattern|None), ...]}"""
    rules: Dict[str, list] = {}

    # ---- EMAIL ----
    rules["EMAIL"] = [
        (re.compile(r"[\w.+-]+@[\w.-]+\.\w{2,}"), None),
    ]

    # ---- PHONE ----
    rules["PHONE"] = [
        (re.compile(r"(?:\+7|8)\s*[\(\[]?\d{3}[\)\]]?\s*\d{3}[\-\s]?\d{2}[\-\s]?\d{2}"), None),
        (re.compile(r"\b\d{10,11}\b"),
         re.compile(r"(?i)телефон|звон|привязан|контакт|смс|sms|мобильн")),
    ]

    # ---- CARD_NUM (16 digits, optionally grouped, NOT followed by =) ----
    rules["CARD_NUM"] = [
        (re.compile(r"\b\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}\b(?!=)"),
         re.compile(r"(?i)карт|card|оплат|списан|баланс|перевод|блокир")),
    ]

    # ---- BANK_ACCT (20 digits, starts with 4/3) ----
    rules["BANK_ACCT"] = [
        (re.compile(r"\b[2-4]\d{19}\b"),
         re.compile(r"(?i)счёт|счет|account|р/с|расчётн|расчетн")),
    ]

    # ---- SNILS ----
    rules["SNILS"] = [
        (re.compile(r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b"), None),
    ]

    # ---- INN (personal 12 / org 10 digits) ----
    rules["INN"] = [
        (re.compile(r"\b\d{12}\b"), re.compile(r"(?i)инн")),
        (re.compile(r"\b\d{10}\b"), re.compile(r"(?i)инн")),
    ]

    # ---- CARD_EXP (MM/YY) ----
    rules["CARD_EXP"] = [
        (re.compile(r"\b(?:0[1-9]|1[0-2])/\d{2}\b"), None),
    ]

    # ---- MAG_STRIPE ----
    rules["MAG_STRIPE"] = [
        (re.compile(r"%B[\w\^/]+\?"), None),
        (re.compile(r"\b9F[0-9A-Fa-f]{2,}[0-9A-Fa-f]+\b"),
         re.compile(r"(?i)emv|магнит|track|полос|операци")),
    ]

    # ---- API_KEY ----
    rules["API_KEY"] = [
        (re.compile(r"\bAIzaSy[\w_-]{30,}"), None),
        (re.compile(r"\bGOCSPX-[\w_-]+"), None),
        (re.compile(r"\bsk_(?:live|test)_[\w]+"), None),
        (re.compile(r"\bpk_(?:live|test)_[\w]+"), None),
        (re.compile(r"\bbk_api_key_[\w]+"), None),
        (re.compile(r"\bdev_key_[\w]+"), None),
        (re.compile(r"\b\d{9,10}:[A-Za-z0-9_-]{30,}"), None),
        (re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"), None),
        (re.compile(r"(?<=['\"])[\w_]+(?=['\"])"),
         re.compile(r"(?i)ключ|key|api|токен|token")),
        (re.compile(r"\b[A-Z][A-Z0-9_]{3,}(?:_[A-Z0-9]+)+\b"),
         re.compile(r"(?i)ключ|key|api|токен|token|документац|удалени|секрет")),
    ]

    # ---- CVV ----
    rules["CVV"] = [
        (re.compile(r"\b\d{3}\b"), re.compile(r"(?i)cvv|cvc")),
    ]

    # ---- PIN ----
    rules["PIN"] = [
        (re.compile(r"\b\d{4}\b"), re.compile(r"(?i)пин|pin")),
    ]

    # ---- OTP ----
    rules["OTP"] = [
        (re.compile(r"\b\d{4,6}\b"),
         re.compile(r"(?i)(?:одноразов|sms|смс).*код|код.*(?:подтвержд|верифик|вход|оплат|sms|смс)|"
                    r"пришёл\s+код|код\s+(?:\d|не\s)")),
    ]

    # ---- DRIVER_LIC ----
    rules["DRIVER_LIC"] = [
        (re.compile(r"\b\d{2}\s\d{2}\s\d{6}\b"),
         re.compile(r"(?i)водител|удостоверени|ву\b")),
        (re.compile(r"\b\d{4}\s?\d{6}\b"),
         re.compile(r"(?i)водител|удостоверени|ву\b")),
        (re.compile(r"\b\d{10}\b"),
         re.compile(r"(?i)водител|удостоверени")),
    ]

    # ---- TEMP_ID ----
    rules["TEMP_ID"] = [
        (re.compile(r"\b[А-ЯA-Z]{2}\d{8,10}\b"),
         re.compile(r"(?i)временн|удостоверени")),
        (re.compile(r"\b\d{6,12}\b"),
         re.compile(r"(?i)временн.*(?:удостоверени|документ)|(?:удостоверени|документ).*временн")),
    ]

    # ---- BIRTH_CERT ----
    rules["BIRTH_CERT"] = [
        (re.compile(r"\b[IVXLCDM]{1,5}-[А-ЯЁ]{2}\s\d{6}\b"), None),
    ]

    # ---- RESIDENCE (вид на жительство) ----
    rules["RESIDENCE"] = [
        (re.compile(r"\b\d{2}[\s№-]+\d{5,7}\b"),
         re.compile(r"(?i)вид на жительств")),
    ]

    # ---- WORK_PERMIT (виза / разрешение на работу) ----
    rules["WORK_PERMIT"] = [
        (re.compile(r"\b\d{2}\s\d{7}\b"),
         re.compile(r"(?i)виз[аыуе]|разрешени.*работ")),
    ]

    # ---- PASSPORT ----
    rules["PASSPORT"] = [
        (re.compile(r"\b\d{2}\s?\d{2}\s\d{6}\b"),
         re.compile(r"(?i)паспорт|серия|номер")),
        (re.compile(
            r"(?:[ОоУу](?:правлени|тдел)\w*\s+)?(?:ОУФМС|УФМС|ФМС|МВД|ГУВД|ОВД|УВД)"
            r"[\w\s,.\-()]*?(?:(?:по|города?|обл(?:асти)?|респ(?:ублик[аие])?|р-на?|района?|край|края|"
            r"[А-ЯЁ][а-яё]+(?:ского|ской|ому|ой)?)\s*)+"
        ), re.compile(r"(?i)паспорт|выда")),
        (re.compile(_DATE_DMY),
         re.compile(r"(?i)паспорт.*выда|выда.*паспорт|выданн")),
        (re.compile(_DATE_TEXT),
         re.compile(r"(?i)паспорт.*выда|выда.*паспорт|выданн")),
    ]

    # ---- PASSWORD ----
    rules["PASSWORD"] = [
        (re.compile(r'(?<=["«])[^\s"«»]{4,}(?=["»])'),
         re.compile(r"(?i)парол")),
        (re.compile(r'(?<=пароль\s)["\s]*\S{4,}'),
         None),
        (re.compile(r'(?<=пароля\s)["\s]*\S{4,}'),
         None),
        (re.compile(r'(?<=например,\s)\S{4,}'),
         re.compile(r"(?i)парол")),
    ]

    # ---- CODE_WORD ----
    rules["CODE_WORD"] = [
        (re.compile(r"(?<=кодовое слово\s)[«\"]?[А-ЯЁа-яё]{2,}[»\"]?"), None),
        (re.compile(r"(?<=кодовое слово — )[«\"]?[А-ЯЁа-яё]{2,}[»\"]?"), None),
        (re.compile(r"(?<=кодовое слово — «)[А-ЯЁа-яё]{2,}(?=»)"), None),
        (re.compile(r"(?<=слово\s)«[А-ЯЁа-яё]{2,}»"),
         re.compile(r"(?i)кодов")),
        (re.compile(r"(?<=«)[А-ЯЁа-яё]{2,}(?=»)"),
         re.compile(r"(?i)кодов\w+\s+слов")),
    ]

    # ---- DOB ----
    rules["DOB"] = [
        (re.compile(_DATE_DMY + r"(?:\s+в\s+\d{1,2}:\d{2}(?::\d{2})?)?"),
         re.compile(r"(?i)рожден|родил")),
        (re.compile(_DATE_TEXT), re.compile(r"(?i)рожден|родил")),
        (re.compile(r"\b(?:19|20)\d{2}\b"),
         re.compile(r"(?i)(?:год|дат)\w*\s+рожден|рожден\w*.*\b\d{4}")),
        (re.compile(_MONTHS_RU, re.I), re.compile(r"(?i)рожден|родил")),
        (re.compile(r"\b\d{1,2}\b"),
         re.compile(r"(?i)(?:рожден|родил).*(?:час|минут)|(?:час|минут).*рожден")),
    ]

    # ---- REG_DATE ----
    rules["REG_DATE"] = [
        (re.compile(_DATE_DMY),
         re.compile(r"(?i)регистрац|пребыван|прописк|местожительств")),
        (re.compile(_DATE_TEXT),
         re.compile(r"(?i)регистрац|пребыван|прописк|местожительств")),
    ]

    # ---- ORG_DATA ----
    rules["ORG_DATA"] = [
        (re.compile(r"\b1\d{12}\b"), re.compile(r"(?i)огрн")),
        (re.compile(r"\b\d{9}\b"), re.compile(r"(?i)кпп|бик")),
        (re.compile(r"\b[34]\d{19}\b"), re.compile(r"(?i)расчётн|расчетн|р/с")),
        (re.compile(r"\b\d{10}\b"),
         re.compile(r"(?i)(?:инн|огрн).*(?:организац|юридич|компан|ооо|зао|пао)|"
                    r"(?:организац|юридич|компан|ооо|зао|пао).*(?:инн|огрн)")),
        (re.compile(
            r"(?:г\.\s?[А-ЯЁ][а-яё]+(?:,?\s*ул\.\s*[А-ЯЁа-яё]+(?:,?\s*д\.\s*\d+)?)?)"
        ), re.compile(r"(?i)юридич.*адрес|адрес.*юридич|организац")),
    ]

    # ---- VEHICLE ----
    rules["VEHICLE"] = [
        (re.compile(r"\b[A-Z0-9]{17}\b"),
         re.compile(r"(?i)vin|автомобил|машин|авто(?:кредит|страхов)")),
        (re.compile(r"\b(?:19|20)\d{2}\b"),
         re.compile(r"(?i)(?:авто|машин|vin).*(?:год|г\.)|\bгод[а]?\s+(?:выпуск|(?:19|20)\d{2})")),
    ]

    return rules


# =====================================================================
#  REGEX DETECTION ENGINE
# =====================================================================

class RegexPIIDetector:
    def __init__(self):
        self.rules = _build_regex_rules()

    def detect(self, text: str) -> List[Tuple[int, int, str]]:
        all_matches = []
        for label, pattern_list in self.rules.items():
            for pat, ctx in pattern_list:
                if ctx is not None and not ctx.search(text):
                    continue
                for m in pat.finditer(text):
                    s, e = m.start(), m.end()
                    if e - s < 1:
                        continue
                    all_matches.append((s, e, label))
        return _resolve_overlaps(all_matches)


def _resolve_overlaps(matches: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    if not matches:
        return []
    sorted_m = sorted(matches, key=lambda x: (x[0], -(x[1] - x[0])))
    result = [sorted_m[0]]
    for m in sorted_m[1:]:
        prev = result[-1]
        if m[0] >= prev[1]:
            result.append(m)
        elif (m[1] - m[0]) > (prev[1] - prev[0]):
            result[-1] = m
    return result


# =====================================================================
#  CUSTOM SPACY COMPONENTS
# =====================================================================

@Language.factory("regex_pii_matcher")
class RegexPIIMatcher:
    """spaCy component: finds PII via regex and stores in doc.spans['regex']."""

    def __init__(self, nlp, name):
        self.detector = RegexPIIDetector()

    def __call__(self, doc: Doc) -> Doc:
        text = doc.text
        matches = self.detector.detect(text)
        spans = []
        for start, end, label in matches:
            span = doc.char_span(start, end, label=label, alignment_mode="expand")
            if span is not None:
                spans.append(span)
        doc.spans["regex"] = spans
        return doc


_QUOTE_CHARS = set('"«»\'"\'')
_REG_DATE_PREFIX = re.compile(r"^(?:с|до|от|по|на|)\s+", re.I)
_ORG_CONTEXT = re.compile(r"(?i)расчётн|расчетн|р/с|организац|юридич|компан|ооо|зао|пао|ао\b")


@Language.factory("entity_merger")
class EntityMerger:
    """Merges doc.spans['regex'] with doc.ents, prioritizing regex matches.
    Also applies post-processing fixes for boundary issues."""

    def __init__(self, nlp, name):
        pass

    def __call__(self, doc: Doc) -> Doc:
        text = doc.text
        regex_spans = list(doc.spans.get("regex", []))
        ner_ents = list(doc.ents)

        all_ents = []
        for span in regex_spans:
            all_ents.append((span.start_char, span.end_char, span.label_, "regex"))
        for ent in ner_ents:
            all_ents.append((ent.start_char, ent.end_char, ent.label_, "ner"))

        merged = _merge_entities(all_ents)
        merged = _postprocess_entities(text, merged)

        final_spans = []
        for start, end, label, _ in merged:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None and len(span) > 0:
                final_spans.append(span)

        try:
            doc.ents = final_spans
        except ValueError:
            doc.ents = _deduplicate_spans(final_spans)
        return doc


def _postprocess_entities(text: str, ents: List[Tuple]) -> List[Tuple]:
    """Fix common boundary issues after merge."""
    result = []
    for start, end, label, source in ents:
        s, e = start, end

        # Strip quotes from PASSWORD and CODE_WORD
        if label in ("PASSWORD", "CODE_WORD"):
            while s < e and text[s] in _QUOTE_CHARS:
                s += 1
            while e > s and text[e - 1] in _QUOTE_CHARS:
                e -= 1
            # Also strip trailing comma/period that got attached
            while e > s and text[e - 1] in (',', '.', ' '):
                e -= 1

        # Strip prepositions from REG_DATE
        if label == "REG_DATE":
            chunk = text[s:e]
            m = _REG_DATE_PREFIX.match(chunk)
            if m and m.group().strip():
                s += m.end()

        # Only reclassify BANK_ACCT→ORG_DATA when "расчётный/расчетный счёт" or "р/с"
        # is directly adjacent (within 15 chars before the entity)
        if label == "BANK_ACCT":
            before = text[max(0, s - 20):s]
            if re.search(r"(?i)расчётн|расчетн|р/с", before):
                label = "ORG_DATA"

        if s < e:
            result.append((s, e, label, source))
    return result


def _merge_entities(ents: List[Tuple]) -> List[Tuple]:
    """Merge entities from regex and NER. Regex has priority."""
    if not ents:
        return []
    priority = {"regex": 0, "ner": 1}
    sorted_e = sorted(ents, key=lambda x: (x[0], -(x[1] - x[0]), priority.get(x[3], 2)))
    result = [sorted_e[0]]
    for e in sorted_e[1:]:
        prev = result[-1]
        if e[0] >= prev[1]:
            result.append(e)
        elif e[3] == "regex" and prev[3] == "ner":
            result[-1] = e
    return result


def _deduplicate_spans(spans: List[Span]) -> List[Span]:
    """Remove overlapping spans keeping the first one."""
    if not spans:
        return []
    sorted_s = sorted(spans, key=lambda s: (s.start, -len(s)))
    result = [sorted_s[0]]
    for s in sorted_s[1:]:
        if s.start >= result[-1].end:
            result.append(s)
    return result


# =====================================================================
#  TRAINING DATA PREPARATION
# =====================================================================

def prepare_examples(nlp, data: List[Dict]) -> List[Example]:
    """Convert dataset to spaCy Example objects for NER training."""
    examples = []
    skipped = 0
    for item in data:
        text = item["text"]
        entities = item["entities"]
        doc = nlp.make_doc(text)
        ents_dict = {"entities": [(s, e, lbl) for s, e, lbl in entities]}
        try:
            example = Example.from_dict(doc, ents_dict)
            examples.append(example)
        except Exception:
            skipped += 1
    if skipped:
        logger.warning("Skipped %d examples with alignment issues", skipped)
    return examples


# =====================================================================
#  PIPELINE BUILD & TRAINING
# =====================================================================

def build_pipeline() -> Language:
    """Build the unified spaCy pipeline."""
    logger.info("Loading ru_core_news_lg base model...")
    nlp = spacy.load("ru_core_news_lg", exclude=["ner"])

    nlp.add_pipe("regex_pii_matcher", before="tok2vec")
    ner = nlp.add_pipe("ner", last=True)
    nlp.add_pipe("entity_merger", last=True)

    for label in ALL_LABELS:
        ner.add_label(label)

    return nlp


def train_pipeline(
    nlp: Language,
    train_data: List[Dict],
    dev_data: List[Dict],
    n_epochs: int = 30,
    batch_size_start: float = 4.0,
    batch_size_end: float = 32.0,
    drop: float = 0.35,
    patience: int = 5,
    output_dir: str = "pii_spacy_model",
):
    """Train the NER component of the pipeline."""
    logger.info("Preparing training examples... (%d train, %d dev)", len(train_data), len(dev_data))

    tok2vec_bytes = nlp.get_pipe("tok2vec").to_bytes()

    train_examples = prepare_examples(nlp, train_data)

    logger.info("Training examples: %d, Dev items: %d", len(train_examples), len(dev_data))

    get_examples = lambda: train_examples[:200]

    with nlp.select_pipes(enable=["tok2vec", "ner"]):
        nlp.initialize(get_examples)

    nlp.get_pipe("tok2vec").from_bytes(tok2vec_bytes)

    optimizer = nlp.create_optimizer()
    best_f1 = 0.0
    no_improve = 0

    for epoch in range(n_epochs):
        random.shuffle(train_examples)
        losses = {}
        batches = minibatch(train_examples, size=compounding(batch_size_start, batch_size_end, 1.001))

        with nlp.select_pipes(enable=["tok2vec", "ner"]):
            for batch in batches:
                nlp.update(batch, sgd=optimizer, losses=losses, drop=drop)

        dev_scores = evaluate_on_data(nlp, dev_data)
        f1 = dev_scores["f1"]
        logger.info(
            "Epoch %02d | loss=%.2f | P=%.3f R=%.3f F1=%.3f",
            epoch, losses.get("ner", 0), dev_scores["precision"], dev_scores["recall"], f1,
        )

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            Path(output_dir).mkdir(exist_ok=True)
            nlp.to_disk(output_dir)
            logger.info("  → New best F1=%.3f, model saved to %s", f1, output_dir)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

    logger.info("Training complete. Best F1=%.3f", best_f1)
    return nlp


# =====================================================================
#  EVALUATION  (strict micro-averaged F1)
# =====================================================================

def extract_entities_from_doc(doc: Doc) -> List[Tuple[int, int, str]]:
    """Extract (start_char, end_char, label) from a processed doc."""
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]


def evaluate_examples(nlp, examples: List[Example]) -> Dict[str, float]:
    """Evaluate on spaCy Example objects."""
    tp = fp = fn = 0
    for example in examples:
        gold_ents = set()
        for ent in example.reference.ents:
            gold_ents.add((ent.start_char, ent.end_char, ent.label_))
        if not gold_ents:
            for s, e, lbl in example.y.user_data.get("entities", []):
                gold_ents.add((s, e, lbl))

        gold_from_data = set()
        try:
            text = example.reference.text
            ents_in_ref = [(ent.start_char, ent.end_char, ent.label_) for ent in example.reference.ents]
            gold_from_data = set(ents_in_ref) if ents_in_ref else gold_ents
        except Exception:
            gold_from_data = gold_ents

        pred_doc = nlp(example.reference.text)
        pred_ents = set(extract_entities_from_doc(pred_doc))

        tp += len(pred_ents & gold_from_data)
        fp += len(pred_ents - gold_from_data)
        fn += len(gold_from_data - pred_ents)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_on_data(nlp, data: List[Dict]) -> Dict[str, float]:
    """Evaluate on raw data dicts (with char offsets)."""
    tp = fp = fn = 0
    per_label_tp = defaultdict(int)
    per_label_fp = defaultdict(int)
    per_label_fn = defaultdict(int)

    for item in data:
        text = item["text"]
        gold = set((s, e, lbl) for s, e, lbl in item["entities"])
        doc = nlp(text)
        pred = set(extract_entities_from_doc(doc))

        matched = pred & gold
        tp += len(matched)
        fp += len(pred - gold)
        fn += len(gold - pred)

        for s, e, lbl in matched:
            per_label_tp[lbl] += 1
        for s, e, lbl in (pred - gold):
            per_label_fp[lbl] += 1
        for s, e, lbl in (gold - pred):
            per_label_fn[lbl] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "per_label_tp": dict(per_label_tp),
        "per_label_fp": dict(per_label_fp),
        "per_label_fn": dict(per_label_fn),
    }


def print_per_label_metrics(scores: Dict):
    """Pretty-print per-label P/R/F1."""
    all_labels = set()
    all_labels.update(scores.get("per_label_tp", {}).keys())
    all_labels.update(scores.get("per_label_fp", {}).keys())
    all_labels.update(scores.get("per_label_fn", {}).keys())

    print(f"\n{'Label':<16} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}  Full Name")
    print("-" * 90)
    for lbl in sorted(all_labels):
        t = scores["per_label_tp"].get(lbl, 0)
        f = scores["per_label_fp"].get(lbl, 0)
        n = scores["per_label_fn"].get(lbl, 0)
        p = t / (t + f) if (t + f) > 0 else 0
        r = t / (t + n) if (t + n) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        full = LABEL_TO_FULL.get(lbl, lbl)
        print(f"{lbl:<16} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {t:>5d} {f:>5d} {n:>5d}  {full}")

    print("-" * 90)
    print(f"{'MICRO':.<16} {scores['precision']:>6.3f} {scores['recall']:>6.3f} {scores['f1']:>6.3f} "
          f"{scores['tp']:>5d} {scores['fp']:>5d} {scores['fn']:>5d}")


# =====================================================================
#  INFERENCE
# =====================================================================

def predict_text(nlp, text: str) -> List[Tuple[int, int, str]]:
    """Predict PII entities and return in competition format (full labels)."""
    doc = nlp(text)
    result = []
    for ent in doc.ents:
        full_label = LABEL_TO_FULL.get(ent.label_, ent.label_)
        result.append((ent.start_char, ent.end_char, full_label))
    return result


def predict_test_set(nlp, test_path: str, output_path: str):
    """Run inference on the private test set and save results."""
    test_data = load_test_data(test_path)
    results = []
    for item in test_data:
        preds = predict_text(nlp, item["text"])
        results.append({
            "id": item["id"],
            "prediction": preds,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Predictions saved to %s (%d rows)", output_path, len(results))
    return results


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="spaCy PII Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on dev set")
    parser.add_argument("--predict", action="store_true", help="Predict on test set")
    parser.add_argument("--regex-only", action="store_true", help="Evaluate regex-only (no NER)")
    parser.add_argument("--data", default="train_dataset.tsv", help="Training data path")
    parser.add_argument("--test", default="private_test_dataset.csv", help="Test data path")
    parser.add_argument("--model-dir", default="pii_spacy_model", help="Model directory")
    parser.add_argument("--output", default="predictions.json", help="Predictions output path")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Dev split ratio")
    parser.add_argument("--full-train", action="store_true",
                        help="Train on full dataset (no dev split) for final submission")
    args = parser.parse_args()

    if args.regex_only:
        logger.info("=== Regex-only evaluation ===")
        data = load_train_data(args.data)
        _, dev_data = split_data(data, args.dev_ratio)
        detector = RegexPIIDetector()
        tp = fp = fn = 0
        per_label_tp = defaultdict(int)
        per_label_fp = defaultdict(int)
        per_label_fn = defaultdict(int)
        for item in dev_data:
            gold = set((s, e, lbl) for s, e, lbl in item["entities"])
            pred = set(tuple(x) for x in detector.detect(item["text"]))
            matched = pred & gold
            tp += len(matched)
            fp += len(pred - gold)
            fn += len(gold - pred)
            for s, e, lbl in matched:
                per_label_tp[lbl] += 1
            for s, e, lbl in (pred - gold):
                per_label_fp[lbl] += 1
            for s, e, lbl in (gold - pred):
                per_label_fn[lbl] += 1
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        scores = {
            "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "per_label_tp": dict(per_label_tp),
            "per_label_fp": dict(per_label_fp),
            "per_label_fn": dict(per_label_fn),
        }
        print_per_label_metrics(scores)
        return

    if args.full_train:
        data = load_train_data(args.data)
        nlp = build_pipeline()
        small_dev = data[:200]
        nlp = train_pipeline(
            nlp, data, small_dev, n_epochs=args.epochs,
            output_dir=args.model_dir, patience=8,
        )
        logger.info("Full-train complete. Model saved to %s", args.model_dir)

    elif args.train:
        data = load_train_data(args.data)
        train_data, dev_data = split_data(data, args.dev_ratio)
        nlp = build_pipeline()
        nlp = train_pipeline(nlp, train_data, dev_data, n_epochs=args.epochs, output_dir=args.model_dir)
        logger.info("=== Final evaluation on dev set ===")
        nlp_best = spacy.load(args.model_dir)
        scores = evaluate_on_data(nlp_best, dev_data)
        print_per_label_metrics(scores)

    if args.evaluate:
        data = load_train_data(args.data)
        _, dev_data = split_data(data, args.dev_ratio)
        nlp = spacy.load(args.model_dir)
        scores = evaluate_on_data(nlp, dev_data)
        print_per_label_metrics(scores)

    if args.predict:
        nlp = spacy.load(args.model_dir)
        predict_test_set(nlp, args.test, args.output)


if __name__ == "__main__":
    main()
