"""Single MATH500 model-performance evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_hf_checkpoint(configured_path: str) -> Path:
    """Resolve an HF checkpoint or the virtual `checkpoints/final` alias."""
    path = _path(configured_path)
    if not path.exists() and path.name == "final":
        candidates = sorted(
            path.parent.glob("global_step_*"),
            key=lambda item: int(item.name.rsplit("_", 1)[-1]),
        )
        if not candidates:
            raise FileNotFoundError(f"no global_step_* checkpoint under {path.parent}")
        path = candidates[-1]
    candidates = [path, path / "actor" / "huggingface", path / "huggingface"]
    for candidate in candidates:
        if (candidate / "config.json").exists() and any(candidate.glob("*.safetensors")):
            return candidate
    raise FileNotFoundError(
        f"{path} is not an HF checkpoint; enable checkpoint.save_hf_model or merge it with `python -m verl.model_merger merge`"
    )


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    if evaluation["benchmark"] != "math500":
        raise ValueError("first release supports evaluation.benchmark=math500 only")
    if not evaluation["checkpoint_path"]:
        raise ValueError("set evaluation.checkpoint_path in YAML before evaluation")

    from datasets import Dataset
    from transformers import AutoTokenizer
    from verl.utils.reward_score.math_reward import compute_score
    from vllm import LLM, SamplingParams

    checkpoint = resolve_hf_checkpoint(evaluation["checkpoint_path"])
    dataset = Dataset.from_parquet(str(_path(config["data"]["val_file"])))
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), trust_remote_code=config["model"]["trust_remote_code"]
    )
    prompts = [render_prompt(tokenizer, row["prompt"]) for row in dataset]
    llm = LLM(
        model=str(checkpoint),
        trust_remote_code=config["model"]["trust_remote_code"],
        tensor_parallel_size=evaluation["tensor_parallel_size"],
        gpu_memory_utilization=evaluation["gpu_memory_utilization"],
        max_model_len=config["data"]["max_prompt_length"] + evaluation["max_response_length"],
        max_num_seqs=evaluation["max_num_seqs"],
        seed=config["experiment"]["seed"],
    )
    sampling = SamplingParams(
        n=evaluation["num_samples_per_prompt"],
        temperature=evaluation["temperature"],
        top_p=evaluation["top_p"],
        top_k=evaluation["top_k"],
        max_tokens=evaluation["max_response_length"],
    )
    outputs = llm.generate(prompts, sampling)
    total_correct = 0
    total_samples = 0
    passed = 0
    total_tokens = 0
    for row, request in zip(dataset, outputs, strict=True):
        ground_truth = row["reward_model"]["ground_truth"]
        correctness = [float(compute_score(item.text, ground_truth)) > 0 for item in request.outputs]
        total_correct += sum(correctness)
        total_samples += len(correctness)
        passed += int(any(correctness))
        total_tokens += sum(len(item.token_ids) for item in request.outputs)
    k = evaluation["num_samples_per_prompt"]
    return {
        "method": config["algorithm"]["name"],
        "checkpoint": str(checkpoint),
        "benchmark": "math500",
        "num_questions": len(dataset),
        "num_samples_per_prompt": k,
        f"acc@{k}": total_correct / total_samples,
        f"pass@{k}": passed / len(dataset),
        "avg_response_length": total_tokens / total_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    results = evaluate(config)
    output_dir = _path(config["experiment"]["output_dir"]) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
