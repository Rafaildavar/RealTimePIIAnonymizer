import pandas as pd
import re

from collections import Counter, defaultdict
from typing import List, Tuple


def extract_entities(text: str) -> List[Tuple[str, int, int]]:
    '''Функция для извлечения сущностей из текста.
    Возвращает список кортежей (сущность, начальная позиция, конечная позиция).'''

    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\[(.*?)]", text)]

def confusion_matrix(target: pd.Series, pred: pd.Series) -> pd.DataFrame:
    """Считает TP/FP/FN для PII-плейсхолдеров
    Сравнение выполняется построчно между target и pred.
    """
    if len(target) != len(pred):
        raise ValueError("target and pred must have the same length")

    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for t, p in zip(target, pred):
        t_text = "" if pd.isna(t) else str(t)
        p_text = "" if pd.isna(p) else str(p)

        target_counts = Counter(entity for entity, _, _ in extract_entities(t_text))
        pred_counts = Counter(entity for entity, _, _ in extract_entities(p_text))

        for entity in set(target_counts) | set(pred_counts):
            tp = min(target_counts[entity], pred_counts[entity])
            fp = max(0, pred_counts[entity] - target_counts[entity])
            fn = max(0, target_counts[entity] - pred_counts[entity])

            stats[entity]["TP"] += tp
            stats[entity]["FP"] += fp
            stats[entity]["FN"] += fn

    result = (
        pd.DataFrame(
            [
                {
                    "entity": entity,
                    "TP": values["TP"],
                    "FP": values["FP"],
                    "FN": values["FN"],
                }
                for entity, values in stats.items()
            ]
        )
        .sort_values("entity")
        .reset_index(drop=True)
    )

    if result.empty:
        return pd.DataFrame(columns=["entity", "TP", "FP", "FN"])

    totals = pd.DataFrame(
        [
            {
                "entity": "__TOTAL__",
                "TP": int(result["TP"].sum()),
                "FP": int(result["FP"].sum()),
                "FN": int(result["FN"].sum()),
            }
        ]
    )

    return pd.concat([result, totals], ignore_index=True)

def metrics(target: pd.Series, pred: pd.Series) -> pd.DataFrame:
    '''Вычисляет Precision, Recall и F1 для каждой сущности на основе confusion matrix.'''

    matrix = confusion_matrix(target, pred)
    df_metrics = pd.DataFrame(matrix['entity'])
    df_metrics['Precision'] = matrix['TP'] / (matrix['TP'] + matrix['FP'])
    df_metrics['Recall'] = matrix['TP'] / (matrix['TP'] + matrix['FN'])
    df_metrics['F1'] = 2 * (df_metrics['Precision'] * df_metrics['Recall']) / (df_metrics['Precision'] + df_metrics['Recall'])

    return df_metrics

if __name__ == '__main__':
    df = pd.read_csv('data/simple_data.csv')
    print(metrics(df.masked, df.masked))
