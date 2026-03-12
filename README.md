# RealTimePIIAnonymizer

Необходима версия Python 11 или 12.

```bash
pip install -r requirements.txt
```

Файл `metrics.py` содержит функции для вычисления метрик Precision, Recall и F1-score.

Файл `masker.py` содержит:
- `mask_pii(text)` - совместимую функцию, возвращающую только замаскированный текст;
- `mask_pii_with_mapping(text)` - возвращает `masked_text` и mapping для обратной подстановки;
- `unmask_pii(masked_text, mapping)` - восстанавливает исходный текст;
- `save_mapping(...)` / `load_mapping(...)` - сохранение и загрузку mapping в JSON.

Для повторяющихся сущностей используются уникальные плейсхолдеры, например: `[EMAIL_1]`, `[EMAIL_2]`, `[NAME_1]`.

[Ссылка на задание](https://event.cu.ru/master-hackathon-ml2026?utm_source=ooh&utm_medium=smm.unp&utm_campaign=cu.events.master%2Fhackathon-ML&utm_term=itmo.master)