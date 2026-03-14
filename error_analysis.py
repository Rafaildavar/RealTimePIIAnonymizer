"""Deep error analysis on dev set for improving F1 to 0.98."""
import spacy
import sys
import re
from collections import defaultdict, Counter

sys.path.insert(0, '.')
from spacy_pii_pipeline import (
    RegexPIIMatcher, EntityMerger, LABEL_TO_FULL, FULL_TO_LABEL,
    load_train_data, split_data, extract_entities_from_doc
)

nlp = spacy.load('pii_spacy_model_v3')
data = load_train_data('train_dataset.tsv')
_, dev_data = split_data(data, 0.2)

tp_total = fp_total = fn_total = 0
per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "fn_examples": [], "fp_examples": []})

for item in dev_data:
    text = item['text']
    gold = set((s, e, lbl) for s, e, lbl in item['entities'])
    doc = nlp(text)
    pred = set(extract_entities_from_doc(doc))

    for ent in gold & pred:
        per_label[ent[2]]["tp"] += 1
    for ent in pred - gold:
        per_label[ent[2]]["fp"] += 1
        if len(per_label[ent[2]]["fp_examples"]) < 8:
            per_label[ent[2]]["fp_examples"].append((text, ent, gold))
    for ent in gold - pred:
        per_label[ent[2]]["fn"] += 1
        if len(per_label[ent[2]]["fn_examples"]) < 8:
            per_label[ent[2]]["fn_examples"].append((text, ent, pred))

    tp_total += len(gold & pred)
    fp_total += len(pred - gold)
    fn_total += len(gold - pred)

p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0
r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0
f1 = 2 * p * r / (p + r) if (p + r) else 0
print(f"\nOVERALL: P={p:.4f} R={r:.4f} F1={f1:.4f}  TP={tp_total} FP={fp_total} FN={fn_total}")
print(f"To reach 0.98 F1, need to fix ~{fp_total + fn_total - int(0.02 * (tp_total + fn_total) * 2)} errors\n")

sorted_labels = sorted(per_label.keys(), key=lambda l: per_label[l]["fp"] + per_label[l]["fn"], reverse=True)

for label in sorted_labels:
    d = per_label[label]
    t, f, n = d["tp"], d["fp"], d["fn"]
    lp = t / (t + f) if (t + f) else 0
    lr = t / (t + n) if (t + n) else 0
    lf = 2 * lp * lr / (lp + lr) if (lp + lr) else 0
    errors = f + n
    if errors == 0:
        continue
    full = LABEL_TO_FULL.get(label, label)
    print(f"{'='*90}")
    print(f"  {label} | F1={lf:.3f} P={lp:.3f} R={lr:.3f} | TP={t} FP={f} FN={n} | errors={errors}")
    print(f"  {full}")
    print(f"{'='*90}")

    if d["fn_examples"]:
        print(f"  MISSED (FN):")
        for text, (s, e, l), preds in d["fn_examples"][:5]:
            ent_text = text[s:e]
            before = text[max(0, s-15):s]
            after = text[e:e+15]
            print(f"    [{before}]>>>{ent_text}<<<[{after}]")
            near_preds = [(ps, pe, pl) for ps, pe, pl in preds if abs(ps - s) < 20 or abs(pe - e) < 20]
            for ps, pe, pl in near_preds:
                print(f"      pred instead: ({ps},{pe},{pl})='{text[ps:pe]}'")

    if d["fp_examples"]:
        print(f"  FALSE POS (FP):")
        for text, (s, e, l), golds in d["fp_examples"][:5]:
            ent_text = text[s:e]
            before = text[max(0, s-15):s]
            after = text[e:e+15]
            print(f"    [{before}]>>>{ent_text}<<<[{after}]")
            near_golds = [(gs, ge, gl) for gs, ge, gl in golds if abs(gs - s) < 20 or abs(ge - e) < 20]
            for gs, ge, gl in near_golds:
                print(f"      gold was: ({gs},{ge},{gl})='{text[gs:ge]}'")
    print()
