"""
Post-process submission2.csv to fix errors and recover missed entities.
Targets: FP removal, FN recovery, label fixes, span boundary fixes.
"""

import csv
import ast
import re
import json
from copy import deepcopy
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional

# ======================================================================
#  CONSTANTS
# ======================================================================

MONTHS_RU = (
    r"(?:январ[яьие]|феврал[яьие]|март[аеу]?|апрел[яьие]|"
    r"ма[яйюе]|июн[яьие]|июл[яьие]|август[аеу]?|"
    r"сентябр[яьие]|октябр[яьие]|ноябр[яьие]|декабр[яьие])"
)
DATE_DMY = r"\d{2}\.\d{2}\.\d{4}"
DATE_TEXT = r"\d{1,2}\s" + MONTHS_RU + r"(?:\s\d{4})?(?:\s?года?)?"

KNOWN_BANKS = {
    'втб', 'сбербанк', 'сбербанке', 'сбербанка', 'альфа-банк', 'альфа-банка',
    'тинькофф', 'тинькофф банк', 'тинькофф банка', 'газпромбанк', 'газпромбанка', 'газпромбанке',
    'россельхозбанк', 'россельхозбанка', 'совкомбанк', 'совкомбанка',
    'открытие', 'банка открытие', 'банке открытие', 'банк открытие', 'банку открытие',
    'почта банк', 'почта банка', 'почта банке', 'мтс банк', 'мтс банка',
    'райффайзенбанк', 'райффайзенбанка', 'росбанк', 'росбанка',
    'ренессанс кредит', 'уралсиб', 'уралсибе', 'бкс банка', 'бкс банке',
    'русский стандарт', 'банка русский стандарт', 'рнкб', 'банк точка', 'банке точка',
    'московского кредитного банка', 'ак барс банка', 'ак барс банке',
    'банка санкт-петербург', 'банком санкт-петербург',
    'дб ао сбербанк', 'ситибанк', 'ситибанке', 'металлинвестбанк', 'металлинвестбанке',
    'восточного банка', 'восточный банк', 'отп банка', 'отп банке',
    'хоум кредит', 'банка хоум кредит', 'промсвязьбанк', 'промсвязьбанка',
    'банк зенит', 'банке зенит', 'банка зенит', 'банка авангард',
    'банка интеза', 'банка урал фд', 'банка кубань кредит', 'банка центр-инвест',
    'бинбанк', 'бинбанка', 'абсолют банка', 'абсолют банке',
    'банка возрождение', 'банке возрождение', 'вбрр', 'банк дом.рф', 'банке дом.рф',
    'halyk bank', 'kaspi bank', 'банка финсервис',
}

KNOWN_CITIES = {
    'москва', 'москве', 'санкт-петербург', 'санкт-петербурге', 'новосибирск', 'новосибирске',
    'екатеринбург', 'екатеринбурге', 'казань', 'казани', 'нижний новгород', 'нижнем новгороде',
    'челябинск', 'челябинске', 'самара', 'самаре', 'омск', 'омске', 'ростов-на-дону', 'ростове-на-дону',
    'уфа', 'уфе', 'красноярск', 'красноярске', 'пермь', 'перми', 'воронеж', 'воронеже',
    'волгоград', 'волгограде', 'краснодар', 'краснодаре', 'саратов', 'саратове',
    'тюмень', 'тюмени', 'тольятти', 'ижевск', 'ижевске', 'барнаул', 'барнауле',
    'ульяновск', 'ульяновске', 'иркутск', 'иркутске', 'хабаровск', 'хабаровске',
    'ярославль', 'ярославле', 'владивосток', 'владивостоке', 'махачкала', 'махачкале',
    'томск', 'томске', 'оренбург', 'оренбурге', 'кемерово', 'новокузнецк', 'новокузнецке',
    'рязань', 'рязани', 'астрахань', 'астрахани', 'набережные челны', 'пенза', 'пензе',
    'липецк', 'липецке', 'тула', 'туле', 'киров', 'кирове', 'чебоксары', 'чебоксарах',
    'калининград', 'калининграде', 'брянск', 'брянске', 'курск', 'курске',
    'иваново', 'улан-удэ', 'белгород', 'белгороде', 'ставрополь', 'ставрополе',
    'владикавказ', 'владикавказе', 'сочи', 'грозный', 'грозном', 'тверь', 'твери',
    'симферополь', 'симферополе', 'архангельск', 'архангельске', 'смоленск', 'смоленске',
    'калуга', 'калуге', 'петрозаводск', 'петрозаводске', 'мурманск', 'мурманске',
    'сургут', 'сургуте', 'нижневартовск', 'нижневартовске', 'череповец', 'череповце',
    'якутск', 'якутске', 'подольск', 'подольске', 'орёл', 'орле', 'вологда', 'вологде',
    'саранск', 'саранске', 'тамбов', 'тамбове', 'стерлитамак', 'стерлитамаке',
    'нижнекамск', 'нижнекамске', 'петропавловск-камчатский',
    'благовещенск', 'благовещенске', 'южно-сахалинск', 'южно-сахалинске',
    'йошкар-ола', 'йошкар-оле', 'сыктывкар', 'сыктывкаре',
    'элиста', 'элисте', 'горно-алтайск', 'абакан', 'абакане', 'кызыл', 'кызыле',
    'магас', 'черкесск', 'черкесске', 'нальчик', 'нальчике', 'майкоп', 'майкопе',
}

KNOWN_COUNTRIES = {
    'россия', 'россии', 'российской', 'сша', 'китай', 'китая', 'китае',
    'германия', 'германии', 'франция', 'франции', 'италия', 'италии', 'италию',
    'испания', 'испании', 'испанию', 'великобритания', 'великобритании',
    'япония', 'японии', 'южная корея', 'южной кореи', 'индия', 'индии',
    'бразилия', 'бразилии', 'канада', 'канады', 'канаде', 'австралия', 'австралии', 'австралию',
    'турция', 'турции', 'турцию', 'египет', 'египте', 'египта',
    'украина', 'украины', 'украину', 'украине', 'белоруссия', 'белоруссии', 'беларусь', 'беларуси',
    'казахстан', 'казахстана', 'казахстане', 'узбекистан', 'узбекистана', 'узбекистане',
    'таджикистан', 'таджикистана', 'киргизия', 'киргизии', 'кыргызстан', 'кыргызстана',
    'грузия', 'грузии', 'азербайджан', 'азербайджана', 'армения', 'армении',
    'молдавия', 'молдавии', 'молдова', 'молдовы', 'латвия', 'латвии',
    'литва', 'литвы', 'эстония', 'эстонии', 'таиланд', 'таиланда', 'таиланде',
    'вьетнам', 'вьетнама', 'мексика', 'мексики', 'мексику',
    'аргентина', 'аргентины', 'аргентине', 'чили', 'перу', 'колумбия', 'колумбии',
    'нидерланды', 'нидерландов', 'нидерландах', 'бельгия', 'бельгии',
    'швейцария', 'швейцарии', 'австрия', 'австрии', 'швеция', 'швеции',
    'норвегия', 'норвегии', 'дания', 'дании', 'финляндия', 'финляндии',
    'польша', 'польши', 'чехия', 'чехии', 'румыния', 'румынии',
    'венгрия', 'венгрии', 'болгария', 'болгарии', 'сербия', 'сербии',
    'хорватия', 'хорватии', 'словения', 'словении', 'словению',
    'португалия', 'португалии', 'греция', 'греции', 'ирландия', 'ирландии',
    'оаэ', 'катар', 'катара', 'саудовская аравия', 'саудовской аравии',
    'израиль', 'израиля', 'израиле', 'марокко', 'тунис', 'тунисе', 'туниса',
    'малайзия', 'малайзии', 'сингапур', 'сингапура', 'сингапуре',
    'индонезия', 'индонезии', 'филиппины', 'филиппин',
    'южная корея', 'южной кореи', 'южную корею',
    'европа', 'европу', 'европы', 'европе', 'азия', 'азии', 'азию',
    'ямало-ненецкий автономный округ', 'чеченская республика',
    'республика татарстан', 'республики татарстан',
    'краснодарский край', 'краснодарского края',
    'московская область', 'московской области',
    'ленинградская область', 'ленинградской области',
    'ростовская область', 'ростовской области',
    'новосибирская область', 'новосибирской области',
    'свердловская область', 'свердловской области',
    'российское', 'испанское', 'итальянское', 'немецкое', 'французское',
    'китайское', 'японское', 'американское', 'британское',
    'кубу', 'куба', 'кубы', 'черногория', 'черногории', 'кипр', 'кипра', 'кипре',
    'мальта', 'мальты', 'мальте', 'исландия', 'исландии',
}


# ======================================================================
#  LOAD DATA
# ======================================================================

def load_test_texts(path: str) -> Dict[int, str]:
    texts = {}
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            texts[int(row['id'])] = row['text']
    return texts


def load_submission(path: str) -> Dict[int, list]:
    preds = {}
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            rid = int(row['id'])
            raw = row['Prediction'].strip()
            if not raw or raw == '[]':
                preds[rid] = []
            else:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, tuple):
                    parsed = [parsed]
                preds[rid] = [list(e) for e in parsed]
    return preds


# ======================================================================
#  REGEX PATTERNS for FN recovery
# ======================================================================

def build_regex_patterns():
    rules = {}

    rules['Email'] = [
        (re.compile(r"[\w.+-]+@[\w.-]+\.\w{2,}"), None),
    ]

    rules['Номер телефона'] = [
        (re.compile(r"(?:\+7|8)\s*[\(\[]?\d{3}[\)\]]?\s*\d{3}[\-\s]?\d{2}[\-\s]?\d{2}"), None),
        (re.compile(r"\b8\s*\(\d{3}\)\s*\d{3}[\-\s]?\d{2}[\-\s]?\d{2}"), None),
        (re.compile(r"\b\d{10,11}\b"),
         re.compile(r"(?i)телефон|звон|привязан|контакт|смс|sms|мобильн")),
    ]

    rules['Номер карты'] = [
        (re.compile(r"\b\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}\b(?!=)"),
         re.compile(r"(?i)карт|card|оплат|списан|баланс|перевод|блокир")),
    ]

    rules['Номер банковского счета'] = [
        (re.compile(r"\b[2-4]\d{19}\b"),
         re.compile(r"(?i)счёт|счет|account|р/с|расчётн|расчетн")),
    ]

    rules['СНИЛС клиента'] = [
        (re.compile(r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b"), None),
    ]

    rules['CVV/CVC'] = [
        (re.compile(r"\b\d{3}\b"), re.compile(r"(?i)cvv|cvc")),
    ]

    rules['ПИН код'] = [
        (re.compile(r"\b\d{4}\b"), re.compile(r"(?i)пин|pin")),
    ]

    rules['Одноразовые коды'] = [
        (re.compile(r"\b\d{4,6}\b"),
         re.compile(r"(?i)(?:одноразов|sms|смс).*код|код.*(?:подтвержд|верифик|вход|оплат|sms|смс)|"
                    r"пришёл\s+код|код\s+(?:\d|не\s)")),
    ]

    rules['Дата окончания срока действия карты'] = [
        (re.compile(r"\b(?:0[1-9]|1[0-2])/\d{2}\b"), None),
        (re.compile(MONTHS_RU + r"\s\d{4}(?:\s?года?)?", re.I),
         re.compile(r"(?i)(?:срок|действ|истека|оконч|до|годен).*(?:карт|card)|"
                    r"(?:карт|card).*(?:срок|действ|истека|оконч|до|годен)")),
        (re.compile(r"\b\d{2}\.\d{2}(?:\.\d{2,4})?\b"),
         re.compile(r"(?i)(?:срок|действ|истека|оконч).*карт|карт.*(?:срок|действ|истека|оконч)")),
    ]

    rules['Содержимое магнитной полосы'] = [
        (re.compile(r"%B[\w\^/]+\?"), None),
        (re.compile(r"\b\d{16}=\d{15,25}\b"), None),
        (re.compile(r"\b9F[0-9A-Fa-f]{2,}[0-9A-Fa-f]+\b"),
         re.compile(r"(?i)emv|магнит|track|полос|чип")),
    ]

    rules['API ключи'] = [
        (re.compile(r"\bAIzaSy[\w_-]{30,}"), None),
        (re.compile(r"\bGOCSPX-[\w_-]+"), None),
        (re.compile(r"\bsk_(?:live|test)_[\w]+"), None),
        (re.compile(r"\bpk_(?:live|test)_[\w]+"), None),
        (re.compile(r"\bbk_api_key_[\w]+"), None),
        (re.compile(r"\bdev_key_[\w]+"), None),
        (re.compile(r"\bAPI_WEBHOOK_\w+"), None),
        (re.compile(r"\b\d{9,10}:[A-Za-z0-9_-]{30,}"), None),
        (re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"), None),
        (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12,}"),
         re.compile(r"(?i)api|ключ|key|токен|token")),
    ]

    rules['Водительское удостоверение'] = [
        (re.compile(r"\b\d{2}\s\d{2}\s\d{6}\b"),
         re.compile(r"(?i)водител|удостоверени|\bву\b")),
        (re.compile(r"\b\d{4}\s?\d{6}\b"),
         re.compile(r"(?i)водител|удостоверени|\bву\b")),
        (re.compile(r"\b\d{10}\b"),
         re.compile(r"(?i)водител|удостоверени")),
    ]

    rules['Временное удостоверение личности'] = [
        (re.compile(r"\b[А-ЯA-Z]{2}\d{8,10}\b"),
         re.compile(r"(?i)временн|удостоверени")),
        (re.compile(r"\b\d{6,12}\b"),
         re.compile(r"(?i)временн.*(?:удостоверени|документ)|(?:удостоверени|документ).*временн")),
    ]

    rules['Свидетельство о рождении'] = [
        (re.compile(r"\b[IVXLCDM]{1,5}-[А-ЯЁ]{2}\s\d{6}\b"), None),
    ]

    rules['Серия и номер вида на жительство'] = [
        (re.compile(r"\b\d{2}[\s№-]+\d{5,7}\b"),
         re.compile(r"(?i)вид на жительств")),
    ]

    rules['Разрешение на работу / визу'] = [
        (re.compile(r"\b\d{2}\s\d{7}\b"),
         re.compile(r"(?i)виз[аыуе]|разрешени.*работ|патент")),
    ]

    rules['Паспортные данные'] = [
        (re.compile(r"\b(\d{2}\s?\d{2})\s+(\d{6})\b"),
         re.compile(r"(?i)паспорт|серия|серии")),
        (re.compile(
            r"(?:[ОоУу](?:правлени|тдел)\w*\s+)?(?:ОУФМС|УФМС|ФМС|МВД|ГУВД|ОВД|УВД|УМВД|ГУМВД)"
            r"[\w\s,.\-()«»]*?(?:(?:по|города?|обл(?:асти)?|респ(?:ублик[аие])?|р-на?|района?|край|"
            r"[А-ЯЁ][а-яё]+(?:ского|ской|скому|ому|ой)?)\s*)+"
        ), re.compile(r"(?i)паспорт|выда")),
        (re.compile(DATE_DMY),
         re.compile(r"(?i)паспорт.*выда|выда.*паспорт|выданн")),
        (re.compile(DATE_TEXT),
         re.compile(r"(?i)паспорт.*выда|выда.*паспорт|выданн")),
        (re.compile(r"\b\d{3}-\d{3}\b"),
         re.compile(r"(?i)код\s+подразделени|паспорт.*код")),
    ]

    rules['Пароли'] = [
        (re.compile(r'(?<=["«])[^\s"«»]{4,}(?=["»])'),
         re.compile(r"(?i)парол")),
        (re.compile(r'(?<=пароль\s)["\s]*\S{4,}'),
         None),
    ]

    rules['Кодовые слова'] = [
        (re.compile(r"(?<=кодовое слово\s)[«\"]?[А-ЯЁа-яё]{2,}[»\"]?"), None),
        (re.compile(r"(?<=кодовое слово — )[«\"]?[А-ЯЁа-яё]{2,}[»\"]?"), None),
        (re.compile(r"(?<=«)[А-ЯЁа-яё]{2,}(?=»)"),
         re.compile(r"(?i)кодов\w+\s+слов")),
    ]

    rules['Сведения об ИНН'] = [
        (re.compile(r"\b\d{12}\b"), re.compile(r"(?i)инн")),
        (re.compile(r"\b\d{10}\b"), re.compile(r"(?i)инн")),
    ]

    rules['Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)'] = [
        (re.compile(r"\b1\d{12}\b"), re.compile(r"(?i)огрн")),
        (re.compile(r"\b\d{9}\b"), re.compile(r"(?i)кпп|бик")),
    ]

    rules['Данные об автомобиле клиента'] = [
        (re.compile(r"\b[A-Z0-9]{17}\b"),
         re.compile(r"(?i)vin|автомобил|машин|авто(?:кредит|страхов)")),
        (re.compile(r"\b[А-Я]\d{3}[А-Я]{2}\d{2,3}\b"),
         re.compile(r"(?i)номер.*авто|гос.*номер|автомобил|машин|транспорт")),
    ]

    rules['Дата рождения'] = [
        (re.compile(DATE_DMY), re.compile(r"(?i)рожден|родил")),
        (re.compile(DATE_TEXT), re.compile(r"(?i)рожден|родил")),
        (re.compile(r"\b(?:19|20)\d{2}\b"),
         re.compile(r"(?i)(?:год|дат)\w*\s+рожден|рожден\w*.*\b\d{4}")),
    ]

    rules['Дата регистрации по месту жительства или пребывания'] = [
        (re.compile(DATE_DMY),
         re.compile(r"(?i)регистрац|пребыван|прописк|местожительств")),
        (re.compile(DATE_TEXT),
         re.compile(r"(?i)регистрац|пребыван|прописк|местожительств")),
    ]

    rules['Имя держателя карты'] = [
        (re.compile(r"\b[A-Z]{2,}\s[A-Z]{2,}\b"),
         re.compile(r"(?i)(?:карт|card|держател|имя на карт|cardholder)")),
    ]

    return rules


# ======================================================================
#  HELPER FUNCTIONS
# ======================================================================

def spans_overlap(s1, e1, s2, e2) -> bool:
    return max(s1, s2) < min(e1, e2)


def is_covered(start, end, entities) -> bool:
    for s, e, _ in entities:
        if spans_overlap(start, end, s, e):
            return True
    return False


def resolve_overlaps(entities: list) -> list:
    if not entities:
        return []
    sorted_e = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    result = [list(sorted_e[0])]
    for e in sorted_e[1:]:
        prev = result[-1]
        if e[0] >= prev[1]:
            result.append(list(e))
        elif (e[1] - e[0]) > (prev[1] - prev[0]):
            result[-1] = list(e)
    return result


# ======================================================================
#  POST-PROCESSING RULES
# ======================================================================

def fix_label_confusion(text: str, entities: list) -> list:
    """Fix common label confusion based on context."""
    text_lower = text.lower()
    result = []

    has_passport_ctx = bool(re.search(r'(?i)паспорт|серия|серии|серию|выда', text))
    has_driver_ctx = bool(re.search(r'(?i)водител|удостоверени.*водител|\bву\b', text))
    has_org_ctx = bool(re.search(
        r'(?i)(?:ООО|ЗАО|ОАО|ПАО)\s*[«"]|АО\s*[«"]|реквизит|контрагент|юридич|организаци|компани',
        text))
    has_inn_word = bool(re.search(r'(?i)\bинн\b', text))
    has_kpp_bik = bool(re.search(r'(?i)\b(?:кпп|бик|огрн)\b', text))

    for ent in entities:
        s, e, label = ent[0], ent[1], ent[2]
        span = text[s:e]
        span_lower = span.lower()
        nearby_before = text[max(0, s - 60):s].lower()
        nearby_after = text[e:min(len(text), e + 60)].lower()
        nearby = nearby_before + ' ' + nearby_after

        # Fix: "Место рождения" → "Наименование банка" when it's a bank name
        if label == 'Место рождения':
            if span_lower in KNOWN_BANKS or any(b in span_lower for b in ['банк', 'кредит']):
                if re.search(r'(?i)банк|кредит|вклад|счёт|счет|карт|перевод', nearby):
                    label = 'Наименование банка'

        # Fix: "Наименование банка" → "Место рождения" when it's a city in birth context
        if label == 'Наименование банка':
            if span_lower in KNOWN_CITIES or re.search(r'(?i)област', span_lower):
                if re.search(r'(?i)родил|рожден|родом', nearby):
                    label = 'Место рождения'

        # Fix: 10-digit number labeled INN → ORG_DATA when org context is present
        if label == 'Сведения об ИНН' and re.match(r'^\d{10}$', span):
            if has_org_ctx or has_kpp_bik:
                if re.search(r'(?i)(?:ООО|ЗАО|ОАО|ПАО|АО)\s*[«"]|реквизит|контрагент', nearby):
                    label = 'Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)'

        # Fix: BANK_ACCT (20 digits) in org context → ORG_DATA
        if label == 'Номер банковского счета':
            if re.search(r'(?i)расчётн|расчетн|р/с', nearby_before):
                label = 'Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)'

        # Fix: "Водительское удостоверение" → "Паспортные данные" when passport context
        if label == 'Водительское удостоверение':
            if has_passport_ctx and not has_driver_ctx:
                label = 'Паспортные данные'

        # Fix: "Одноразовые коды" → "Паспортные данные" for 6-digit number in passport context
        if label == 'Одноразовые коды' and re.match(r'^\d{6}$', span):
            if has_passport_ctx and re.search(r'(?i)паспорт|серия|номер', nearby_before):
                label = 'Паспортные данные'

        # Fix: "СНИЛС клиента" with wrong digit count → "Паспортные данные" (код подразделения)
        if label == 'СНИЛС клиента':
            digits = re.sub(r'\D', '', span)
            if len(digits) != 11:
                if has_passport_ctx and re.search(r'(?i)код|подразделени', nearby):
                    label = 'Паспортные данные'

        # Fix: "Место рождения" in passport context for a city name
        # (this is actually OK - birthplace cities can appear in passport context)

        # Fix: "Имя держателя карты" for non-name-like values
        if label == 'Имя держателя карты':
            if not re.match(r'^[A-Z\s]+$', span) and not re.match(r'^[А-ЯЁа-яё\s]+$', span):
                if re.search(r'(?i)автомобил|авто|vin|машин', nearby):
                    label = 'Данные об автомобиле клиента'

        # Fix: "Пароли" for common non-password words
        if label == 'Пароли':
            if span_lower in ('должен', 'менее', 'не менее', 'включая', 'содержать',
                              'символов', 'заглавные', 'строчные', 'цифры', 'минимум'):
                continue  # skip this entity entirely

        # Fix: date labeled as wrong category
        if label == 'Дата регистрации по месту жительства или пребывания':
            if re.search(r'(?i)(?:срок|действ|истека|оконч|годен).*карт|карт.*(?:срок|действ|истека)', nearby):
                label = 'Дата окончания срока действия карты'

        if label == 'Дата окончания срока действия карты':
            if re.search(r'(?i)регистрац|пребыван|прописк', nearby):
                if not re.search(r'(?i)карт', nearby):
                    label = 'Дата регистрации по месту жительства или пребывания'

        result.append([s, e, label])
    return result


def fix_span_boundaries(text: str, entities: list) -> list:
    """Fix common span boundary issues."""
    result = []
    for ent in entities:
        s, e, label = ent[0], ent[1], ent[2]
        span = text[s:e]

        # Strip leading spaces
        while s < e and text[s] == ' ':
            s += 1

        # Strip trailing commas/spaces (but NOT periods/!/? which may be part of data)
        safe_strip = {' ', ','}
        if label not in ('Содержимое магнитной полосы', 'Пароли', 'API ключи'):
            while e > s and text[e-1] in safe_strip:
                e -= 1

        # Strip quotes from CODE_WORD
        if label == 'Кодовые слова':
            while s < e and text[s] in '"«\'""':
                s += 1
            while e > s and text[e-1] in '"»\'""':
                e -= 1

        # API_KEY: strip trailing sentence punctuation
        if label == 'API ключи':
            while e > s and text[e-1] in ('.', ',', ';', ':', ' '):
                e -= 1

        if s < e:
            result.append([s, e, label])
    return result


def remove_false_positives(text: str, entities: list) -> list:
    """Remove clearly incorrect predictions."""
    text_lower = text.lower()
    result = []

    for ent in entities:
        s, e, label = ent[0], ent[1], ent[2]
        if s < 0 or e > len(text) or s >= e:
            continue

        span = text[s:e]

        # Remove "МВД. Мы" type errors (span crosses a REAL sentence boundary)
        # But NOT abbreviations like "г.", "обл.", "д.", "ул.", "пр-т."
        skip_dot_check = {'Полный адрес', 'API ключи', 'Содержимое магнитной полосы',
                          'Паспортные данные',
                          'Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)'}
        if '. ' in span and label not in skip_dot_check:
            for m_dot in re.finditer(r'\. ', span):
                dot_pos = m_dot.start()
                before_dot = span[:dot_pos].rstrip()
                after_dot = span[dot_pos+2:]
                if before_dot and before_dot[-1] not in 'гдулкпс' and after_dot and after_dot[0].isupper():
                    e = s + dot_pos
                    span = text[s:e]
                    break
            if e <= s:
                continue

        # Remove: "30 дней" / "3 месяца" as REG_DATE (duration, not a date)
        if label == 'Дата регистрации по месту жительства или пребывания':
            if re.match(r'^\d+\s+(?:дн|день|мес|недел|лет|год)', span, re.I):
                continue

        # Remove: "Тарифы" / common words as BIRTHPLACE
        if label == 'Место рождения':
            if span_lower := span.lower():
                if span_lower in ('тарифы', 'тариф', 'услуги', 'продукты', 'справка',
                                  'выписка', 'документы', 'условия', 'информация'):
                    continue

        # Remove: single char Temp ID
        if label == 'Временное удостоверение личности':
            digits = re.sub(r'\D', '', span)
            if len(digits) < 6:
                continue

        # Remove: phone with < 7 digits (unless strong phone context)
        if label == 'Номер телефона':
            digits = re.sub(r'\D', '', span)
            if len(digits) < 7:
                nearby_ctx = text[max(0,s-40):min(len(text),e+40)].lower()
                if not re.search(r'телефон|звонок|номер|горяч|линия|контакт', nearby_ctx):
                    continue

        result.append([s, e, label])
    return result


def recover_missing_entities(text: str, entities: list, rules: dict) -> list:
    """Use regex patterns to find entities missed by the model."""
    recovered = list(entities)

    for label, patterns in rules.items():
        for pat, ctx in patterns:
            if ctx is not None and not ctx.search(text):
                continue
            for m in pat.finditer(text):
                ms, me = m.start(), m.end()
                if me - ms < 2:
                    continue
                if not is_covered(ms, me, [(e[0], e[1], e[2]) for e in recovered]):
                    recovered.append([ms, me, label])

    return resolve_overlaps(recovered)


# ======================================================================
#  MAIN
# ======================================================================

def main():
    print("Loading data...")
    test_texts = load_test_texts('private_test_dataset.csv')
    preds = load_submission('submission2.csv')
    regex_rules = build_regex_patterns()

    print(f"Loaded {len(test_texts)} test texts, {len(preds)} predictions")

    total_before = sum(len(v) for v in preds.values())
    changes = {'added': 0, 'removed': 0, 'label_changed': 0, 'boundary_changed': 0}

    improved = {}
    for rid in sorted(test_texts.keys()):
        text = test_texts[rid]
        ents = preds.get(rid, [])
        ents_before = [tuple(e) for e in ents]

        # Step 1: Fix label confusion
        ents = fix_label_confusion(text, ents)

        # Step 2: Fix span boundaries
        ents = fix_span_boundaries(text, ents)

        # Step 3: Remove false positives
        ents = remove_false_positives(text, ents)

        # Step 4: Recover missing entities via regex
        ents = recover_missing_entities(text, ents, regex_rules)

        # Step 5: Final overlap resolution
        ents = resolve_overlaps(ents)

        # Track changes
        ents_after = [tuple(e) for e in ents]
        for e in ents_after:
            if e not in ents_before:
                changes['added'] += 1
        for e in ents_before:
            if e not in ents_after:
                changes['removed'] += 1

        improved[rid] = ents

    total_after = sum(len(v) for v in improved.values())

    print(f"\nEntities before: {total_before}")
    print(f"Entities after:  {total_after}")
    print(f"Added: {changes['added']}, Removed: {changes['removed']}")

    # Save
    output_path = 'submission2_improved.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'Prediction'])
        for rid in sorted(improved.keys()):
            ents = improved[rid]
            if not ents:
                writer.writerow([rid, '[]'])
            else:
                pred_str = str([(s, e, l) for s, e, l in ents])
                writer.writerow([rid, pred_str])

    print(f"Saved to {output_path}")

    # Show sample changes
    print("\n=== SAMPLE CHANGES (first 20) ===")
    shown = 0
    for rid in sorted(test_texts.keys()):
        old = set(tuple(e) for e in preds.get(rid, []))
        new = set(tuple(e) for e in improved[rid])
        if old != new:
            shown += 1
            if shown <= 20:
                text = test_texts[rid]
                only_old = old - new
                only_new = new - old
                print(f"\nid={rid}:")
                for s, e, l in sorted(only_old):
                    sp = text[s:e] if 0 <= s < e <= len(text) else "OOB"
                    print(f"  - REMOVED: [{s}:{e}] \"{sp[:60]}\" → {l}")
                for s, e, l in sorted(only_new):
                    sp = text[s:e] if 0 <= s < e <= len(text) else "OOB"
                    print(f"  + ADDED:   [{s}:{e}] \"{sp[:60]}\" → {l}")

    print(f"\nTotal rows changed: {shown}")


if __name__ == '__main__':
    main()
