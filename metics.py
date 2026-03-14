import ast
import pandas as pd
import re

from typing import Iterable, Set, Tuple


Entity = Tuple[int, int, str]


def _to_entity_set(value) -> Set[Entity]:
    """Нормализует вход (строка/список) в множество сущностей (start, end, category)."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()

    parsed = value
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw == "[]":
            return set()
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            # Fallback для редких случаев поломанного сериализованного списка.
            pattern = r"\((\d+)\s*,\s*(\d+)\s*,\s*['\"](.*?)['\"]\)"
            return {
                (int(start), int(end), str(label))
                for start, end, label in re.findall(pattern, raw)
            }

    if isinstance(parsed, tuple):
        parsed = [parsed]

    if not isinstance(parsed, Iterable) or isinstance(parsed, (str, bytes)):
        return set()

    entities: Set[Entity] = set()
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        start, end, label = item
        try:
            entities.add((int(start), int(end), str(label)))
        except (TypeError, ValueError):
            continue

    return entities


def confusion_matrix(target: pd.Series, pred: pd.Series) -> pd.DataFrame:
    """Считает TP/FP/FN по strict matching: start, end, category.

    Возвращает DataFrame со строкой на каждую категорию + итоговую строку __TOTAL__.
    """

    if len(target) != len(pred):
        raise ValueError("target and pred must have the same length")

    from collections import defaultdict

    stats: dict = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for gold_raw, pred_raw in zip(target, pred):
        gold_entities = _to_entity_set(gold_raw)
        pred_entities = _to_entity_set(pred_raw)

        for entity in gold_entities & pred_entities:
            stats[entity[2]]["TP"] += 1

        for entity in pred_entities - gold_entities:
            stats[entity[2]]["FP"] += 1

        for entity in gold_entities - pred_entities:
            stats[entity[2]]["FN"] += 1

    rows = [
        {"entity": cat, "TP": v["TP"], "FP": v["FP"], "FN": v["FN"]}
        for cat, v in sorted(stats.items())
    ]

    total_tp = sum(r["TP"] for r in rows)
    total_fp = sum(r["FP"] for r in rows)
    total_fn = sum(r["FN"] for r in rows)
    rows.append({"entity": "__TOTAL__", "TP": total_tp, "FP": total_fp, "FN": total_fn})

    return pd.DataFrame(rows)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics(target: pd.Series, pred: pd.Series) -> pd.DataFrame:
    """Вычисляет Precision/Recall/F1 по каждой категории и итоговые значения."""

    matrix = confusion_matrix(target, pred)

    rows = []
    for _, row in matrix.iterrows():
        tp, fp, fn = int(row["TP"]), int(row["FP"]), int(row["FN"])
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        rows.append({
            "entity": row["entity"],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Пример из условия: одна сущность совпала, одна - с неверной категорией.
    gold = pd.Series([
        "[(71, 75, 'Паспортные данные РФ'), (82, 88, 'Паспортные данные РФ')]"
    ])
    pred = pd.Series([
        "[(71, 75, 'Паспортные данные РФ'), (82, 88, 'ИНН')]"
    ])

    print(metrics(gold, pred))
