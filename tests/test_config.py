from copy import deepcopy
from pathlib import Path

from train import load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def _controlled(config):
    value = deepcopy(config)
    value["experiment"].pop("name")
    value["experiment"].pop("output_dir")
    value["algorithm"].pop("name")
    value["algorithm"].pop("adv_estimator")
    value["algorithm"].pop("rl_zvp", None)
    value["entropy"].pop("calculate")
    return value


def test_formal_configs_are_controlled_comparison():
    grpo = load_config(ROOT / "configs/grpo_qwen3_1.7b_math.yaml")
    rl_zvp = load_config(ROOT / "configs/rl_zvp_qwen3_1.7b_math.yaml")
    validate_config(grpo)
    validate_config(rl_zvp)
    assert _controlled(grpo) == _controlled(rl_zvp)


def test_debug_configs_validate_and_are_controlled():
    grpo = load_config(ROOT / "configs/debug_grpo.yaml")
    rl_zvp = load_config(ROOT / "configs/debug_rl_zvp.yaml")
    validate_config(grpo)
    validate_config(rl_zvp)
    assert _controlled(grpo) == _controlled(rl_zvp)


def test_paper_batch_mapping_is_explicit():
    config = load_config(ROOT / "configs/grpo_qwen3_1.7b_math.yaml")
    assert config["data"]["train_batch_size"] == 512
    assert config["rollout"]["num_samples_per_prompt"] == 8
    assert config["training"]["prompt_mini_batch_size"] == 32
    assert 32 * 8 == 256
    assert 512 // 32 == 16
    assert config["training"]["total_gradient_steps"] // 16 == 42
