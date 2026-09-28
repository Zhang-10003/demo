"""Prepare MATH train and MATH500 parquet files in verl's native schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def extract_ground_truth(example: dict[str, Any]) -> str:
    if example.get("answer") not in (None, ""):
        return str(example["answer"])
    solution = str(example.get("solution", ""))
    from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

    boxed = last_boxed_only_string(solution)
    if boxed is None:
        raise ValueError("example has neither `answer` nor a boxed solution")
    return remove_boxed(boxed)


def convert_split(dataset, *, split: str, data_source: str, prompt_suffix: str) -> Dataset:
    rows = []
    for index, example in enumerate(dataset):
        question = str(example.get("problem", example.get("question", ""))).strip()
        if not question:
            raise ValueError(f"empty question at {split}[{index}]")
        ground_truth = extract_ground_truth(example)
        prompt = f"{question} {prompt_suffix}".strip()
        rows.append(
            {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "question": question,
                    "answer": ground_truth,
                },
            }
        )
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_cfg = config["data"]
    train_raw = load_dataset(data_cfg["train_source"], split="train")
    eval_raw = load_dataset(data_cfg["eval_source"], split="test")
    train = convert_split(
        train_raw,
        split="train",
        data_source=data_cfg["train_source"],
        prompt_suffix=data_cfg["prompt_suffix"],
    )
    math500 = convert_split(
        eval_raw,
        split="test",
        data_source=data_cfg["eval_source"],
        prompt_suffix=data_cfg["prompt_suffix"],
    )
    train_path = Path(data_cfg["train_file"])
    eval_path = Path(data_cfg["val_file"])
    train_path = train_path if train_path.is_absolute() else PROJECT_ROOT / train_path
    eval_path = eval_path if eval_path.is_absolute() else PROJECT_ROOT / eval_path
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(str(train_path))
    math500.to_parquet(str(eval_path))
    print(f"wrote {len(train)} train rows to {train_path}")
    print(f"wrote {len(math500)} MATH500 rows to {eval_path}")


if __name__ == "__main__":
    main()
