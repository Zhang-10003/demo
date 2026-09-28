"""Finite-sampling all-wrong diagnosis; no training, entropy or algorithm changes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import warnings

import yaml

from evaluate import render_prompt

PROJECT_ROOT = Path(__file__).resolve().parent
INITIAL_COLUMNS = ["example_id", "category", "level", "num_samples", "correct_count",
                   "wrong_count", "accuracy", "reward_mean", "reward_variance", "group_type"]
RECOVERY_COLUMNS = ["example_id", "category", "level", "initial_correct_count",
                    "extra_num_samples", "extra_correct_count", "extra_wrong_count",
                    "p_hat", "recoverable", "prob_initial_all_wrong_under_phat"]


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def prompt_suffix(config: dict) -> str:
    """Read the existing reproduction YAML; there is no second template."""
    with project_path(config["data"]["prompt_config"]).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["data"]["prompt_suffix"]


def load_config(path: str) -> dict:
    with project_path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    # Explicit required fields avoid invisible experimental defaults.
    required = {
        "model": ("name_or_path", "trust_remote_code", "dtype"),
        "data": ("root", "processed_dir", "split", "max_samples", "prompt_config",
                 "expected_categories", "expected_train_size", "future_train_size",
                 "diagnostic_size", "pilot_size", "future_train_file", "diagnostic_file", "pilot_file"),
        "rollout": ("initial_group_size", "recovery_num_samples", "temperature", "top_p", "top_k",
                    "max_response_length", "max_prompt_length", "tensor_parallel_size",
                    "gpu_memory_utilization", "max_num_seqs"),
        "diagnosis": ("resume", "batch_size", "top_case_count", "case_correct_responses", "case_wrong_responses"),
        "output": ("dir",),
    }
    if type(cfg["seed"]) is not int:
        raise ValueError("seed must be an integer")
    for section, fields in required.items():
        for field in fields:
            if field not in cfg[section]:
                raise ValueError(f"missing {section}.{field}")
    data, rollout = cfg["data"], cfg["rollout"]
    for section, fields in (("data", ("expected_train_size", "future_train_size", "diagnostic_size", "pilot_size")),
                            ("rollout", ("initial_group_size", "recovery_num_samples", "max_response_length",
                                         "max_prompt_length", "tensor_parallel_size", "max_num_seqs")),
                            ("diagnosis", ("batch_size", "top_case_count", "case_correct_responses", "case_wrong_responses"))):
        for field in fields:
            if type(cfg[section][field]) is not int or cfg[section][field] <= 0:
                raise ValueError(f"{section}.{field} must be a positive integer")
    if data["future_train_size"] + data["diagnostic_size"] != data["expected_train_size"]:
        raise ValueError("future_train + diagnostic must equal original train size")
    if not data["pilot_size"] <= data["diagnostic_size"]:
        raise ValueError("pilot must fit inside diagnostic")
    if data["max_samples"] is not None and (type(data["max_samples"]) is not int or data["max_samples"] <= 0):
        raise ValueError("max_samples must be null or a positive integer")
    if data["split"] not in (data["diagnostic_file"], data["pilot_file"]):
        raise ValueError("diagnosis may use only the diagnostic or pilot split")
    if rollout["temperature"] <= 0 or not 0 < rollout["top_p"] <= 1:
        raise ValueError("stochastic sampling requires temperature > 0 and 0 < top_p <= 1")
    if type(rollout["top_k"]) is not int or rollout["top_k"] not in (-1,) and rollout["top_k"] <= 0:
        raise ValueError("top_k must be -1 or a positive integer")
    if not 0 < rollout["gpu_memory_utilization"] < 1:
        raise ValueError("invalid gpu_memory_utilization")
    if type(cfg["diagnosis"]["resume"]) is not bool:
        raise ValueError("resume must be boolean")
    return cfg


def sample_key(example_id: str, stage: str, sample_id: int) -> tuple:
    return example_id, stage, sample_id


def sampling_seed(seed: int, key: tuple) -> int:
    """Stable independent RNG stream per sample; unaffected by resume/batch order."""
    payload = json.dumps([seed, *key], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def classify_group(correct: list[int]) -> str:
    if not correct or any(value not in (0, 1) for value in correct):
        raise ValueError("group requires nonempty binary correctness")
    return "all_wrong" if not any(correct) else "all_correct" if all(correct) else "mixed"


def compute_recovery_statistics(extra_correct_count: int, extra_num_samples: int, initial_group_size: int) -> dict:
    if extra_num_samples <= 0 or initial_group_size <= 0 or not 0 <= extra_correct_count <= extra_num_samples:
        raise ValueError("invalid recovery counts")
    p_hat = extra_correct_count / extra_num_samples
    return {"extra_num_samples": extra_num_samples, "extra_correct_count": extra_correct_count,
            "extra_wrong_count": extra_num_samples - extra_correct_count, "p_hat": p_hat,
            "recoverable": extra_correct_count > 0,
            "prob_initial_all_wrong_under_phat": (1 - p_hat) ** initial_group_size}


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_rollouts(path: Path, stage: str, ids: set[str], count: int) -> dict:
    """Reject duplicate/foreign keys; back up and discard only a torn final line."""
    records = {}
    if not path.exists():
        return records
    with path.open("rb") as handle:
        number = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            number += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if line.endswith(b"\n") or handle.read(1):
                    raise ValueError(f"invalid JSONL at {path}:{number}; refusing to discard a complete/interior line") from error
                backup = path.with_name(path.name + f".torn-tail-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}")
                backup.write_bytes(line)
                with path.open("r+b") as repair:
                    repair.truncate(offset)
                warnings.warn(f"Interrupted final JSONL line backed up to {backup}; missing sample will be regenerated", RuntimeWarning)
                break
            key = sample_key(record["example_id"], record["stage"], record["sample_id"])
            if key in records:
                raise ValueError(f"duplicate rollout key at {path}:{number}: {key}")
            if key[0] not in ids or key[1] != stage or type(key[2]) is not int or not 0 <= key[2] < count:
                raise ValueError(f"unexpected rollout key: {key}")
            if type(record["correct"]) is not bool or not math.isfinite(record["reward"]):
                raise ValueError(f"invalid correctness/reward: {key}")
            records[key] = record
            if not line.endswith(b"\n"):
                # Valid final JSON can survive a crash before its newline.
                with path.open("ab") as repair:
                    repair.write(b"\n")
    return records


def assert_complete(rows: list[dict], stage: str, count: int, records: dict):
    expected = {sample_key(row["example_id"], stage, i) for row in rows for i in range(count)}
    if set(records) != expected:
        raise ValueError(f"{stage} incomplete: missing={len(expected - set(records))}, extra={len(set(records) - expected)}")


def run_rollouts(rows, stage, count, config, llm, prompts, records, path, sampling_class, verifier):
    """Only missing keys are submitted. Write and fsync after each prompt batch.

    n=1 requests use distinct seeds; no RNG stream is reset across responses.
    Gold answers are passed only to the verifier after generation.
    """
    rollout = config["rollout"]
    batch_size = config["diagnosis"]["batch_size"]
    pending = [row for row in rows if any(sample_key(row["example_id"], stage, i) not in records for i in range(count))]
    with path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            jobs, params = [], []
            for row in pending[start:start + batch_size]:
                for i in range(count):
                    key = sample_key(row["example_id"], stage, i)
                    if key in records:
                        continue
                    seed = sampling_seed(config["seed"], key)
                    jobs.append((row, key, seed))
                    params.append(sampling_class(n=1, temperature=rollout["temperature"],
                        top_p=rollout["top_p"], top_k=rollout["top_k"],
                        max_tokens=rollout["max_response_length"], seed=seed))
            outputs = llm.generate([prompts[row["example_id"]] for row, _, _ in jobs], params, use_tqdm=False)
            for (row, key, seed), request in zip(jobs, outputs, strict=True):
                if len(request.outputs) != 1:
                    raise ValueError("expected exactly one response per seeded request")
                answer = request.outputs[0]
                reward = float(verifier(answer.text, row["ground_truth"]))
                if not math.isfinite(reward):
                    raise ValueError(f"non-finite verifier reward: {key}")
                record = {"example_id": key[0], "stage": stage, "sample_id": key[2],
                    "category": row["category"], "level": row["level"], "problem": row["problem"],
                    "prompt": prompts[key[0]], "response": answer.text, "reward": reward,
                    "correct": reward > 0, "generated_token_count": len(answer.token_ids),
                    "finish_reason": answer.finish_reason, "temperature": rollout["temperature"],
                    "top_p": rollout["top_p"], "top_k": rollout["top_k"], "sampling_seed": seed,
                    "model_path": config["model"]["name_or_path"]}
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                records[key] = record
            os.fsync(handle.fileno())
            completed = [row for row in rows if all(sample_key(row["example_id"], stage, i) in records for i in range(count))]
            if stage == "initial":
                wrong = sum(not any(records[sample_key(row["example_id"], stage, i)]["correct"] for i in range(count)) for row in completed)
                print(f"[Initial] Prompts completed: {len(completed)}/{len(rows)}; Responses generated: {len(records)}/{len(rows)*count}; All-wrong discovered: {wrong}", flush=True)
            else:
                print(f"[Recovery] All-wrong prompts: {len(rows)}; Completed: {len(completed)}/{len(rows)}", flush=True)
    assert_complete(rows, stage, count, records)


def run_initial_rollouts(rows, config, **kwargs):
    return run_rollouts(rows, "initial", config["rollout"]["initial_group_size"], config, **kwargs)


def run_recovery_rollouts(rows, config, **kwargs):
    return run_rollouts(rows, "recovery", config["rollout"]["recovery_num_samples"], config, **kwargs)


def initial_summary(rows, records, group_size):
    summary = []
    for row in rows:
        samples = [records[sample_key(row["example_id"], "initial", i)] for i in range(group_size)]
        correct = [int(sample["correct"]) for sample in samples]
        rewards = [sample["reward"] for sample in samples]
        summary.append({key: row[key] for key in ("example_id", "category", "level")} | {
            "num_samples": group_size, "correct_count": sum(correct), "wrong_count": group_size - sum(correct),
            "accuracy": sum(correct)/group_size, "reward_mean": statistics.mean(rewards),
            "reward_variance": statistics.pvariance(rewards), "group_type": classify_group(correct)})
    return summary


def breakdown(initial, recovery, field):
    recovered = {row["example_id"]: row for row in recovery}
    groups = defaultdict(list)
    for row in initial:
        groups[row[field]].append(row)
    lines = [f"| {field} | num_prompts | initial_all_wrong_count | initial_all_wrong_rate | recoverable_count | recoverable_rate | mean_p_hat |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for value, rows in sorted(groups.items()):
        wrong = [recovered[row["example_id"]] for row in rows if row["group_type"] == "all_wrong"]
        count = sum(row["recoverable"] for row in wrong)
        rate = f"{count/len(wrong):.4f}" if wrong else "N/A"
        mean = f"{statistics.mean(row['p_hat'] for row in wrong):.4f}" if wrong else "N/A"
        lines.append(f"| {value} | {len(rows)} | {len(wrong)} | {len(wrong)/len(rows):.4f} | {count} | {rate} | {mean} |")
    return "\n".join(lines)


def write_report(output, manifest, initial, recovery, config):
    g, k = config["rollout"]["initial_group_size"], config["rollout"]["recovery_num_samples"]
    n, m = len(recovery), sum(row["recoverable"] for row in recovery)
    rates = f"{m}/{n} = {m/n:.4%}" if n else "N/A (no initial all-wrong prompts)"
    lines = ["# All-wrong finite-sampling diagnosis", "", "## Dataset statistics", ""]
    lines += [f"- {name}: {value}" for name, value in manifest["counts"].items()]
    lines += ["", f"## Initial G={g}", "", f"Total prompts: {len(initial)}", ""]
    for kind in ("all_wrong", "mixed", "all_correct"):
        count = sum(row["group_type"] == kind for row in initial)
        lines.append(f"- {kind}: {count}/{len(initial)} ({count/len(initial):.2%})")
    lines += ["", f"## Recovery: additional K={k}", "", f"Initial all-wrong prompts: {n}",
              f"Recoverable: {m}", f"No correct answer observed in recovery: {n-m}",
              f"Recoverable rate: {rates}", "", "## p_hat distribution (initial all-wrong only)", ""]
    values = [row["p_hat"] for row in recovery]
    if values:
        lines += [f"- {name}: {value:.6f}" for name, value in {
            "mean": statistics.mean(values), "median": statistics.median(values),
            "std (population)": statistics.pstdev(values), "min": min(values), "max": max(values)}.items()]
        for label, predicate in (("p_hat = 0", lambda p: p == 0),
            ("0 < p_hat <= 0.05", lambda p: 0 < p <= .05),
            ("0.05 < p_hat <= 0.10", lambda p: .05 < p <= .10),
            ("0.10 < p_hat <= 0.25", lambda p: .10 < p <= .25),
            ("p_hat > 0.25", lambda p: p > .25)):
            count = sum(predicate(p) for p in values)
            lines.append(f"- {label}: {count}/{n} ({count/n:.2%})")
    else:
        lines.append("N/A: empty all-wrong subset.")
    lines += ["", "## By category", "", breakdown(initial, recovery, "category"),
              "", "## By difficulty", "", breakdown(initial, recovery, "level"), "",
              "## Interpretation", "",
              "Recovery shows nonzero observed success under additional sampling. Failure to recover does not establish zero success probability or zero learnability.",
              "The plug-in probability (1-p_hat)^G assumes independent draws at a fixed success probability. p_hat uses recovery samples only and is uncertain at finite K; it is not proof of the true probability.",
              "No GO/KILL threshold is inferred automatically; inspect the measured rate and sample count."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cases(output, rows, recovery, initial_records, recovery_records, config):
    by_id = {row["example_id"]: row for row in rows}
    cfg = config["diagnosis"]
    cases = sorted((row for row in recovery if row["recoverable"]),
                   key=lambda row: (-row["p_hat"], row["example_id"]))[:cfg["top_case_count"]]
    lines = ["# Top recoverable initial all-wrong cases", ""]
    if not cases:
        lines.append("No recoverable case observed.")
    for case in cases:
        row = by_id[case["example_id"]]
        initial = [initial_records[sample_key(row["example_id"], "initial", i)] for i in range(config["rollout"]["initial_group_size"])]
        extra = [recovery_records[sample_key(row["example_id"], "recovery", i)] for i in range(config["rollout"]["recovery_num_samples"])]
        lines += [f"## {row['example_id']}", "", f"Category: {row['category']}; Level: {row['level']}",
                  "", row["problem"], "", f"Initial rewards: {[sample['reward'] for sample in initial]}",
                  f"Extra correct count: {case['extra_correct_count']}; p_hat: {case['p_hat']}", ""]
        selected = [("Initial", sample) for sample in initial]
        selected += [("Recovery correct", sample) for sample in [s for s in extra if s["correct"]][:cfg["case_correct_responses"]]]
        selected += [("Recovery wrong", sample) for sample in [s for s in extra if not s["correct"]][:cfg["case_wrong_responses"]]]
        for label, sample in selected:
            lines += [f"### {label} sample {sample['sample_id']} (finish: {sample['finish_reason']})", "", sample["response"], ""]
        lines += ["### Gold solution (analysis only; never sent to model)", "", row["solution"], ""]
    (output / "top_recoverable_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose(config):
    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from verl.utils.reward_score.math_reward import compute_score
    from vllm import LLM, SamplingParams

    data, rollout = config["data"], config["rollout"]
    processed = project_path(data["processed_dir"])
    manifest = json.loads((processed / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest["seed"] != config["seed"] or manifest["prompt_suffix"] != prompt_suffix(config):
        raise ValueError("split seed/prompt differs from YAML; prepare matching splits first")
    if manifest["data_root"] != str(project_path(data["root"])):
        raise ValueError("split source root differs from YAML")
    expected_counts = {"original_train": data["expected_train_size"], "future_train": data["future_train_size"],
                       "diagnostic": data["diagnostic_size"], "pilot": data["pilot_size"]}
    if manifest["counts"] != expected_counts:
        raise ValueError("prepared split sizes differ from YAML")
    split_path = processed / data["split"]
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != manifest["parquet_sha256"][data["split"]]:
        raise ValueError("prepared split was modified; refusing inconsistent diagnosis data")
    all_rows = list(Dataset.from_parquet(str(split_path)))
    split_key = "pilot" if data["split"] == data["pilot_file"] else "diagnostic"
    if sorted(row["example_id"] for row in all_rows) != manifest["split_ids"][split_key]:
        raise ValueError("split IDs differ from preparation manifest")
    rows = all_rows if data["max_samples"] is None else all_rows[:data["max_samples"]]
    if not rows:
        raise ValueError("empty diagnosis dataset")
    suffix = prompt_suffix(config)
    for row in rows:
        expected = [{"role": "user", "content": f"{row['problem']} {suffix}".strip()}]
        if row["prompt"] != expected:
            raise ValueError(f"unexpected prompt template: {row['example_id']}")
    output = project_path(config["output"]["dir"])
    output.mkdir(parents=True, exist_ok=True)
    # Linux flock releases on crash; concurrent jobs cannot append to one run.
    import fcntl
    lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("another diagnosis process owns this output directory")
    try:
        _diagnose_locked(config, rows, manifest, split_path, output, pd, torch,
                         AutoTokenizer, LLM, SamplingParams, compute_score)
    finally:
        lock.close()


def _diagnose_locked(config, rows, manifest, split_path, output, pd, torch, tokenizer_class, llm_class, sampling_class, verifier):
    rollout = config["rollout"]
    identity = json.loads(json.dumps(config))
    identity["diagnosis"].pop("resume")
    identity["resolved_prompt_suffix"] = prompt_suffix(config)
    identity["split_sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    identity["seed_scheme"] = "sha256(base_seed,example_id,stage,sample_id)/63bit/n=1/v1"
    metadata_path = output / "metadata.json"
    packages = {}
    for name in ("torch", "transformers", "vllm", "verl"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    if metadata_path.exists():
        if not config["diagnosis"]["resume"]:
            raise ValueError("output already exists and resume=false; choose a new output.dir")
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved["run_identity"] != identity:
            raise ValueError("config/data/prompt changed; refusing to mix experiments")
        if saved["versions"] != packages:
            raise ValueError("library versions changed; choose a new output.dir for reproducibility")
    else:
        if any((output / name).exists() for name in ("initial_rollouts.jsonl", "recovery_rollouts.jsonl")):
            raise ValueError("rollout files exist without metadata; cannot establish run identity")
        write_json(metadata_path, {"run_identity": identity, "config": config,
            "model_path": config["model"]["name_or_path"], "dataset_path": str(split_path),
            "seed": config["seed"], "python_version": platform.python_version(),
            "versions": packages, "cuda_version": torch.version.cuda,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "timestamp": datetime.now(timezone.utc).isoformat()})
    print(yaml.safe_dump({"model": config["model"]["name_or_path"], "dataset": config["data"]["root"],
        "diagnosis_split": str(split_path), "number_of_prompts": len(rows),
        "seed": config["seed"], **rollout}, sort_keys=False), flush=True)
    ids = {row["example_id"] for row in rows}
    g, k = rollout["initial_group_size"], rollout["recovery_num_samples"]
    initial_path, recovery_path = output / "initial_rollouts.jsonl", output / "recovery_rollouts.jsonl"
    initial = load_rollouts(initial_path, "initial", ids, g)
    # Do not load a GPU model when a completed run only needs report regeneration.
    llm = None
    prompts = {}
    def generate(stage_rows, stage, count, records, path):
        nonlocal llm, prompts
        expected = len(stage_rows) * count
        if len(records) == expected:
            assert_complete(stage_rows, stage, count, records)
            path.touch(exist_ok=True)
            print(f"[{stage}] Already complete: {len(stage_rows)} prompts; no new generation.")
            return
        if llm is None:
            if torch.cuda.device_count() < rollout["tensor_parallel_size"]:
                raise ValueError("visible GPU count is smaller than tensor_parallel_size")
            tokenizer = tokenizer_class.from_pretrained(config["model"]["name_or_path"],
                trust_remote_code=config["model"]["trust_remote_code"])
            prompts = {row["example_id"]: render_prompt(tokenizer, row["prompt"]) for row in rows}
            for row in rows:
                if len(tokenizer.encode(prompts[row["example_id"]], add_special_tokens=False)) > rollout["max_prompt_length"]:
                    raise ValueError(f"prompt exceeds max_prompt_length; no silent truncation: {row['example_id']}")
            llm = llm_class(model=config["model"]["name_or_path"],
                trust_remote_code=config["model"]["trust_remote_code"], dtype=config["model"]["dtype"],
                tensor_parallel_size=rollout["tensor_parallel_size"],
                gpu_memory_utilization=rollout["gpu_memory_utilization"],
                max_model_len=rollout["max_prompt_length"] + rollout["max_response_length"],
                max_num_seqs=rollout["max_num_seqs"], seed=config["seed"])
        runner = run_initial_rollouts if stage == "initial" else run_recovery_rollouts
        runner(stage_rows, config, llm=llm, prompts=prompts, records=records, path=path,
               sampling_class=sampling_class, verifier=verifier)
    generate(rows, "initial", g, initial, initial_path)
    summary = initial_summary(rows, initial, g)
    pd.DataFrame(summary, columns=INITIAL_COLUMNS).to_parquet(output / "initial_summary.parquet", index=False)
    wrong_ids = {row["example_id"] for row in summary if row["group_type"] == "all_wrong"}
    wrong_rows = [row for row in rows if row["example_id"] in wrong_ids]
    recovery = load_rollouts(recovery_path, "recovery", wrong_ids, k)
    print(f"[Recovery] Initial all-wrong prompts: {len(wrong_rows)}", flush=True)
    generate(wrong_rows, "recovery", k, recovery, recovery_path)
    assert_complete(wrong_rows, "recovery", k, recovery)
    recovery_summary = []
    for row in wrong_rows:
        correct = sum(recovery[sample_key(row["example_id"], "recovery", i)]["correct"] for i in range(k))
        recovery_summary.append({key: row[key] for key in ("example_id", "category", "level")} |
            {"initial_correct_count": 0} | compute_recovery_statistics(correct, k, g))
    pd.DataFrame(recovery_summary, columns=RECOVERY_COLUMNS).to_parquet(output / "recovery_summary.parquet", index=False)
    write_report(output, manifest, summary, recovery_summary, config)
    write_cases(output, rows, recovery_summary, initial, recovery, config)
    print(f"Report: {output / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    diagnose(load_config(args.config))


if __name__ == "__main__":
    main()
