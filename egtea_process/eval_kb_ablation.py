"""Grouped evaluation for the B-group KB ablation (Table 6 of the plan).

For every kb_mode it reports overall / inside-K0 / outside-K0:

    noun_strict     Object accuracy, Top-1
    verb_strict     Action accuracy, Top-1
    phrase_strict   Phrase accuracy, Top-1 verb AND noun (== trans.py action_correct_top1)
    phrase_at_k     Phrase accuracy under the looser Top-k rule (reference only)

Note: in this evaluation "Action" and "Phrase" are numerically identical because the
repository defines a correct action as ``verb correct AND noun correct``. Do not put
both columns in the paper as if they were independent metrics.

K0 is read from ``context_base/pre_noun.json`` (the 388 initial concepts).

Usage:
    python egtea_process/eval_kb_ablation.py --pred_dir ablation_kb
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trans import (  # noqa: E402
    get_verb_id,
    load_noun_mapping,
    load_egtea_labels,
    parse_predictions,
)

DEFAULT_LABEL = "/mnt/data/anonymous/mydata/EGTEA++/EGTEA/Action_Annotations/test_split1.txt"
DEFAULT_NOUN_MAP = "/mnt/data/anonymous/mydata/EGTEA++/EGTEA/Action_Annotations/noun_idx.txt"
DEFAULT_K0 = "/mnt/data/anonymous/mycode/action_agent_vllm/context_base/pre_noun.json"


def normalize_noun(text):
    return str(text).strip().lower()


def load_k0(profile_file):
    with open(profile_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {normalize_noun(item.get("key", "")) for item in data if item.get("key")}


def load_id_to_noun(noun_map_file):
    """Reverse noun mapping, kept local so this script does not depend on nltk."""
    mapping = {}
    with open(noun_map_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().rsplit(maxsplit=1)
            if len(parts) == 2:
                word, idx = parts
                mapping[int(idx)] = word.strip().lower()
    return mapping


def aggregate(records):
    n = len(records)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for key in records[0]:
        out[key] = 100.0 * sum(r[key] for r in records) / n
    return out


def eval_one(result_file, label_file, noun_map_file, k0_file, top_k=5):
    noun_map = load_noun_mapping(noun_map_file)
    id_to_noun = load_id_to_noun(noun_map_file)
    gt = load_egtea_labels(label_file)
    k0 = load_k0(k0_file)

    if not os.path.exists(result_file):
        return None

    predictions = parse_predictions(result_file, top_k=top_k)
    buckets = {"all": [], "inside": [], "outside": []}

    for vid, pred in predictions.items():
        if vid not in gt:
            continue

        gt_verb_id, gt_noun_ids = gt[vid]
        gt_nouns = [normalize_noun(id_to_noun.get(i, "")) for i in gt_noun_ids]
        gt_nouns = [n for n in gt_nouns if n]
        if not gt_nouns:
            continue

        pred_verbs = pred["verbs"]
        pred_nouns = pred["nouns"]
        selected = pred["selected_nouns"]

        # ---- strict Top-1 ----
        pred_verb_id = get_verb_id(pred_verbs[0]) if pred_verbs else -1
        verb_ok = int(pred_verb_id == gt_verb_id and pred_verb_id != -1)

        if len(gt_nouns) == 1:
            noun_ok = int(bool(pred_nouns) and noun_map.get(pred_nouns[0], -1) == gt_noun_ids[0])
        else:
            sel_ids = {noun_map.get(n, -1) for n in selected} - {-1}
            noun_ok = int(set(gt_noun_ids).issubset(sel_ids))

        # ---- loose Top-k ----
        pred_verb_ids = {get_verb_id(v) for v in pred_verbs[:top_k]} - {-1}
        verb_ok_k = gt_verb_id in pred_verb_ids
        if len(gt_nouns) == 1:
            pred_noun_ids = {noun_map.get(n, -1) for n in pred_nouns[:top_k]} - {-1}
            noun_ok_k = gt_noun_ids[0] in pred_noun_ids
        else:
            sel_ids = {noun_map.get(n, -1) for n in selected} - {-1}
            noun_ok_k = set(gt_noun_ids).issubset(sel_ids)

        record = {
            "noun_strict": noun_ok,
            "verb_strict": verb_ok,
            "phrase_strict": int(verb_ok and noun_ok),
            "phrase_at_k": int(verb_ok_k and noun_ok_k),
        }
        buckets["all"].append(record)

        # A clip counts as "inside" only if every GT noun already exists in K0.
        inside = all(noun in k0 for noun in gt_nouns)
        buckets["inside" if inside else "outside"].append(record)

    return {name: aggregate(recs) for name, recs in buckets.items()}


def main():
    parser = argparse.ArgumentParser(description="Grouped evaluation of the KB ablation")
    parser.add_argument("--pred_dir", type=str, default="ablation_kb")
    parser.add_argument("--modes", type=str, default="none,static,dynamic,persistent")
    parser.add_argument("--label_file", type=str, default=DEFAULT_LABEL)
    parser.add_argument("--noun_map_file", type=str, default=DEFAULT_NOUN_MAP)
    parser.add_argument("--k0_file", type=str, default=DEFAULT_K0)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    summary = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        result_file = os.path.join(args.pred_dir, f"{mode}_results.txt")
        result = eval_one(
            result_file, args.label_file, args.noun_map_file, args.k0_file, args.top_k
        )
        if result is None:
            print(f"[skip] {result_file} not found")
            continue
        summary[mode] = result

    header = f"{'config':<12}{'bucket':<9}{'n':>6}{'Phrase':>9}{'Action':>9}{'Object':>9}{'Phrase@k':>10}"
    print(header)
    print("-" * len(header))
    for mode, buckets in summary.items():
        for bucket in ("all", "inside", "outside"):
            agg = buckets.get(bucket, {"n": 0})
            if not agg.get("n"):
                continue
            print(
                f"{mode:<12}{bucket:<9}{agg['n']:>6}"
                f"{agg['phrase_strict']:>9.2f}{agg['verb_strict']:>9.2f}"
                f"{agg['noun_strict']:>9.2f}{agg['phrase_at_k']:>10.2f}"
            )
        print("-" * len(header))

    out_file = os.path.join(args.pred_dir, "summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"summary -> {out_file}")


if __name__ == "__main__":
    main()
