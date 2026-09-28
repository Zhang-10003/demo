"""Scientific bookkeeping tests; fake rollout tests do not measure model accuracy."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose as d
from data.prepare_diagnosis import stratified_take


@pytest.mark.parametrize("values, expected", [
    ([0, 0, 0, 0], "all_wrong"), ([1, 1, 1, 1], "all_correct"), ([0, 1, 0, 1], "mixed")])
def test_group_classification(values, expected):
    assert d.classify_group(values) == expected


def test_recovery_statistics():
    zero = d.compute_recovery_statistics(0, 32, 8)
    assert zero["recoverable"] is False
    recovered = d.compute_recovery_statistics(4, 32, 8)
    assert recovered["recoverable"] is True
    assert recovered["p_hat"] == .125
    assert recovered["extra_wrong_count"] == 28
    assert recovered["prob_initial_all_wrong_under_phat"] == pytest.approx(.875 ** 8)


def test_resume_keys_and_seeds():
    keys = [d.sample_key("example", stage, i) for stage in ("initial", "recovery") for i in range(32)]
    assert len(set(keys)) == 64
    seeds = [d.sampling_seed(42, key) for key in keys]
    assert len(set(seeds)) == len(seeds)
    assert seeds == [d.sampling_seed(42, key) for key in keys]


def test_stratified_partition_exact_reproducible():
    rows = [{"example_id": str(i), "category": f"c{i % 7}", "level": f"Level {i % 5 + 1}"} for i in range(7500)]
    # Also exercise singleton strata, which sklearn stratification can reject.
    rows[0]["level"] = "singleton"
    diagnostic, future = stratified_take(rows, 500, 42)
    pilot, _ = stratified_take(diagnostic, 100, 42)
    ids = lambda group: {row["example_id"] for row in group}
    assert (len(future), len(diagnostic), len(pilot)) == (7000, 500, 100)
    assert not ids(future) & ids(diagnostic)
    assert ids(future) | ids(diagnostic) == ids(rows)
    assert ids(pilot) <= ids(diagnostic)
    assert (diagnostic, future) == stratified_take(list(reversed(rows)), 500, 42)


def record(example="a", stage="initial", i=0, correct=False):
    return {"example_id": example, "stage": stage, "sample_id": i, "correct": correct, "reward": float(correct)}


def test_resume_duplicate_foreign_fail_and_torn_tail_recovers(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    line = json.dumps(record()) + "\n"
    path.write_text(line + line)
    with pytest.raises(ValueError, match="duplicate"):
        d.load_rollouts(path, "initial", {"a"}, 2)
    path.write_text(json.dumps(record(stage="recovery")))
    with pytest.raises(ValueError, match="unexpected"):
        d.load_rollouts(path, "initial", {"a"}, 2)
    path.write_text(line + '{"example_id":')
    with pytest.warns(RuntimeWarning, match="backed up"):
        records = d.load_rollouts(path, "initial", {"a"}, 2)
    assert len(records) == 1
    assert path.read_text() == line
    assert len(list(tmp_path.glob("*.torn-tail-*"))) == 1
    path.write_text(line + '{broken}\n')
    with pytest.raises(ValueError, match="invalid JSONL"):
        d.load_rollouts(path, "initial", {"a"}, 2)


def test_only_missing_samples_generated(tmp_path):
    config = d.load_config("configs/debug_diagnosis.yaml")
    rows = [{"example_id": "a", "category": "algebra", "level": "Level 1", "problem": "problem", "ground_truth": "SECRET GOLD"}]
    path = tmp_path / "initial.jsonl"
    existing = record()
    path.write_text(json.dumps(existing) + "\n")
    records = d.load_rollouts(path, "initial", {"a"}, 2)
    calls = []
    class FakeLLM:
        def generate(self, prompts, params, **kwargs):
            assert prompts == ["problem only"]
            assert len(params) == 1
            assert params[0].seed == d.sampling_seed(config["seed"], ("a", "initial", 1))
            assert params[0].n == 1
            calls.append(1)
            return [SimpleNamespace(outputs=[SimpleNamespace(text="correct", token_ids=[1, 2], finish_reason="stop")])]
    verifier = lambda text, gold: 1 if text == "correct" and gold == "SECRET GOLD" else 0
    kwargs = dict(llm=FakeLLM(), prompts={"a": "problem only"}, records=records,
                  path=path, sampling_class=SimpleNamespace, verifier=verifier)
    d.run_initial_rollouts(rows, config, **kwargs)
    d.run_initial_rollouts(rows, config, **kwargs)
    assert calls == [1]
    assert len(d.load_rollouts(path, "initial", {"a"}, 2)) == 2


def test_full_fake_pipeline_and_completed_resume(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    config = d.load_config("configs/debug_diagnosis.yaml")
    suffix = d.prompt_suffix(config)
    rows = [{"example_id": name, "category": "algebra", "level": "Level 1",
             "problem": name, "solution": "SECRET SOLUTION", "ground_truth": "SECRET GOLD",
             "prompt": [{"role": "user", "content": f"{name} {suffix}"}]} for name in ("wrong", "mixed", "correct")]
    split = tmp_path / "split.parquet"
    pd.DataFrame(rows).to_parquet(split)
    calls = []
    class Tokenizer:
        chat_template = None
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def encode(self, *args, **kwargs):
            return [1, 2]
    class FakeLLM:
        def __init__(self, **kwargs):
            pass
        def generate(self, prompts, params, **kwargs):
            recovery = bool(calls)
            calls.append(prompts)
            outputs = []
            for position, prompt in enumerate(prompts):
                assert "SECRET" not in prompt
                if recovery:
                    assert prompt.startswith("user: wrong ")
                    correct = position == 0
                else:
                    correct = prompt.startswith("user: correct ") or (prompt.startswith("user: mixed ") and position % 2 == 0)
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(
                    text="yes" if correct else "no", token_ids=[1], finish_reason="stop")]))
            return outputs
    torch = SimpleNamespace(version=SimpleNamespace(cuda="fake"), cuda=SimpleNamespace(
        device_count=lambda: 1, get_device_name=lambda i: "fake"))
    manifest = {"counts": {"original_train": 7500, "future_train": 7000, "diagnostic": 500, "pilot": 100}}
    verifier = lambda text, gold: float(text == "yes")
    d._diagnose_locked(config, rows, manifest, split, tmp_path, pd, torch, Tokenizer, FakeLLM, SimpleNamespace, verifier)
    assert len(calls) == 2
    initial = pd.read_parquet(tmp_path / "initial_summary.parquet")
    assert initial["group_type"].tolist() == ["all_wrong", "mixed", "all_correct"]
    recovery = pd.read_parquet(tmp_path / "recovery_summary.parquet")
    assert recovery["example_id"].tolist() == ["wrong"]
    assert recovery["extra_num_samples"].tolist() == [4]
    assert recovery["p_hat"].tolist() == [.25]
    assert "Recoverable rate: 1/1" in (tmp_path / "report.md").read_text()
    assert "SECRET SOLUTION" in (tmp_path / "top_recoverable_cases.md").read_text()
    assert len((tmp_path / "initial_rollouts.jsonl").read_text().splitlines()) == 6
    assert len((tmp_path / "recovery_rollouts.jsonl").read_text().splitlines()) == 4
    class NoModel:
        def __init__(self, **kwargs):
            raise AssertionError("completed resume must not load model")
    d._diagnose_locked(config, rows, manifest, split, tmp_path, pd, torch, Tokenizer, NoModel, SimpleNamespace, verifier)
    changed = copy.deepcopy(config)
    changed["rollout"]["temperature"] = .9
    with pytest.raises(ValueError, match="changed"):
        d._diagnose_locked(changed, rows, manifest, split, tmp_path, pd, torch, Tokenizer, NoModel, SimpleNamespace, verifier)


def test_no_all_wrong_report(tmp_path):
    config = d.load_config("configs/debug_diagnosis.yaml")
    manifest = {"counts": {"original_train": 7500}}
    initial = [{"example_id": "a", "category": "algebra", "level": "Level 1", "group_type": "mixed"}]
    d.write_report(tmp_path, manifest, initial, [], config)
    assert "N/A (no initial all-wrong prompts)" in (tmp_path / "report.md").read_text()


def test_diagnosis_uses_existing_prompt_and_formal_sampling():
    config = d.load_config("configs/diagnosis_qwen3_1.7b_math.yaml")
    import yaml
    baseline = yaml.safe_load(d.project_path(config["data"]["prompt_config"]).read_text())
    assert d.prompt_suffix(config) == baseline["data"]["prompt_suffix"]
    assert config["rollout"]["initial_group_size"] == baseline["rollout"]["num_samples_per_prompt"]
    for key in ("temperature", "top_p", "top_k"):
        assert config["rollout"][key] == baseline["rollout"][key]
    assert config["rollout"]["max_response_length"] == baseline["data"]["max_response_length"]
