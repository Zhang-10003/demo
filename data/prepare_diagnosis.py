"""Split only local MATH *train* into future train / diagnostic / nested pilot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def stratified_take(rows: list[dict], count: int, seed: int):
    """Exact-size deterministic proportional allocation by category + level.

    Largest remainders handle small strata without sklearn's minimum-size
    restrictions. A singleton can land in either split; every row is retained.
    """
    if not 0 <= count <= len(rows):
        raise ValueError("split size exceeds available rows")
    if not rows:
        return [], []
    strata = defaultdict(list)
    for row in rows:
        strata[(row["category"], row["level"])].append(row)
    keys = sorted(strata)
    quotas = {key: count * len(strata[key]) // len(rows) for key in keys}
    order = sorted(keys, key=lambda key: (-(count * len(strata[key]) % len(rows)), key))
    for key in order[:count - sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(seed)
    selected, remaining = [], []
    for key in keys:
        group = sorted(strata[key], key=lambda row: row["example_id"])
        rng.shuffle(group)
        selected.extend(group[:quotas[key]])
        remaining.extend(group[quotas[key]:])
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def extract_diagnosis_ground_truth(original, example_id, overrides, extractor):
    """Reuse the existing extractor; repair only explicit, verified YAML entries.

    Never infer an answer from an arbitrary empty box. A configured repair must
    match the source solution and must not replace an already valid annotation.
    """
    gold = extractor(original)
    if example_id in overrides:
        override = overrides[example_id]
        expected = override["expected_solution"]
        if " ".join(str(original["solution"]).split()) != " ".join(expected.split()):
            raise ValueError(f"answer override source mismatch: {example_id}")
        if gold not in (None, ""):
            raise ValueError(f"answer override would replace a nonempty annotation: {example_id}")
        gold = override["ground_truth"]
        if not isinstance(gold, str) or not gold.strip() or not override["reason"].strip():
            raise ValueError(f"invalid answer override: {example_id}")
        print(f"[Annotation correction] {example_id}: empty -> {gold!r}; {override['reason']}", flush=True)
        return gold, "explicit_yaml_override"
    if gold in (None, ""):
        raise ValueError(f"empty extracted answer: {example_id}; inspect the source and add a verified data.answer_overrides entry")
    return gold, "existing_extractor"


def load_hendrycks_math_train(config: dict) -> tuple[list[dict], list[dict]]:
    import pyarrow.parquet as pq
    from data.prepare_math import extract_ground_truth
    from diagnose import project_path, prompt_suffix

    cfg = config["data"]
    root = project_path(cfg["root"])
    # This pattern never touches test-*.parquet, even for schema inspection.
    files = sorted(root.glob("*/train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no */train-*.parquet found under {root}")
    categories = sorted({path.parent.name for path in files})
    if categories != sorted(cfg["expected_categories"]):
        raise ValueError(f"unexpected train categories: {categories}")
    suffix = prompt_suffix(config)
    rows, sources = [], []
    overrides = cfg.get("answer_overrides", {})
    applied_overrides = set()
    for path in files:
        schema = pq.read_schema(path)
        print(f"Train parquet: {path}\nSchema:\n{schema}", flush=True)
        for index, original in enumerate(pq.read_table(path).to_pylist()):
            problem = str(original.get("problem", "")).strip()
            if not problem or not original.get("solution"):
                raise ValueError(f"missing problem/solution: {path}:{index}")
            category = path.parent.name
            level = str(original.get("level") or "unknown")
            # File name disambiguates rows when a category has multiple shards.
            example_id = f"{category}/{path.name}:{index}"
            gold, answer_source = extract_diagnosis_ground_truth(original, example_id, overrides, extract_ground_truth)
            if answer_source == "explicit_yaml_override":
                applied_overrides.add(example_id)
            rows.append({
                "example_id": example_id, "category": category, "level": level,
                "problem": problem, "solution": str(original["solution"]),
                "gold_answer": gold, "ground_truth": gold,
                "gold_answer_source": answer_source,
                "prompt": [{"role": "user", "content": f"{problem} {suffix}".strip()}],
                "data_source": "hendrycks_math", "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gold},
                "extra_info": {"split": "train", "index": index, "category": category,
                               "level": level, "example_id": example_id},
            })
        sources.append({"path": str(path), "schema": str(schema),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if applied_overrides != set(overrides):
        raise ValueError(f"unused answer overrides: {sorted(set(overrides) - applied_overrides)}")
    if len(rows) != cfg["expected_train_size"]:
        raise ValueError(f"expected {cfg['expected_train_size']} train rows, found {len(rows)}")
    print(f"Train files: {len(files)}; examples: {len(rows)}; categories: {categories}")
    print(f"Levels: {dict(sorted(Counter(row['level'] for row in rows).items()))}")
    return rows, sources


def prepare_diagnosis_split(config: dict) -> dict:
    from datasets import Dataset
    from diagnose import project_path, prompt_suffix, write_json

    rows, sources = load_hendrycks_math_train(config)
    cfg = config["data"]
    diagnostic, future = stratified_take(rows, cfg["diagnostic_size"], config["seed"])
    pilot, _ = stratified_take(diagnostic, cfg["pilot_size"], config["seed"])
    original_ids = {row["example_id"] for row in rows}
    diagnostic_ids = {row["example_id"] for row in diagnostic}
    future_ids = {row["example_id"] for row in future}
    assert len(original_ids) == len(rows), "duplicate original IDs"
    assert not diagnostic_ids & future_ids
    assert diagnostic_ids | future_ids == original_ids
    assert {row["example_id"] for row in pilot} <= diagnostic_ids
    assert len(future) == cfg["future_train_size"]
    output = project_path(cfg["processed_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": config["seed"], "data_root": str(project_path(cfg["root"])),
        "sources": sources, "prompt_suffix": prompt_suffix(config),
        "answer_overrides_applied": cfg.get("answer_overrides", {}),
        "files": {key: cfg[key] for key in ("future_train_file", "diagnostic_file", "pilot_file")},
        "allocation": "category+level proportional largest remainder; sorted strata + seeded shuffle",
        "counts": {"original_train": len(rows), "future_train": len(future),
                   "diagnostic": len(diagnostic), "pilot": len(pilot)},
        "categories": sorted({row["category"] for row in rows}),
        "levels": dict(sorted(Counter(row["level"] for row in rows).items())),
        "split_ids": {"future_train": sorted(future_ids), "diagnostic": sorted(diagnostic_ids),
                      "pilot": sorted(row["example_id"] for row in pilot)},
    }
    manifest_path = output / "split_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_identity = {key: value for key, value in previous.items() if key != "parquet_sha256"}
        if previous_identity != manifest:
            raise ValueError("existing split manifest differs; choose another processed_dir")
        if all((output / cfg[key]).exists() for key in
               ("future_train_file", "diagnostic_file", "pilot_file")):
            for name, expected_hash in previous["parquet_sha256"].items():
                if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected_hash:
                    raise ValueError(f"prepared split was modified: {output / name}")
            print("Existing identical splits retained.")
            return previous
    for key, split in (("future_train_file", future), ("diagnostic_file", diagnostic),
                       ("pilot_file", pilot)):
        target = output / cfg[key]
        temporary = target.with_suffix(".parquet.tmp")
        Dataset.from_list(split).to_parquet(str(temporary))
        temporary.replace(target)
        print(f"Wrote {len(split)} examples: {target}")
    manifest["parquet_sha256"] = {
        cfg[key]: hashlib.sha256((output / cfg[key]).read_bytes()).hexdigest()
        for key in ("future_train_file", "diagnostic_file", "pilot_file")
    }
    write_json(manifest_path, manifest)
    return manifest


def main():
    from diagnose import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare_diagnosis_split(load_config(args.config))


if __name__ == "__main__":
    main()
