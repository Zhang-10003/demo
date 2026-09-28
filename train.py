"""Single thin training entrypoint: YAML -> pinned verl v1 trainer."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_VERL_VERSION = "0.9.1"
EXPECTED_VERL_COMMIT = "1876b06d0a3e4e71e06230be10af14492ca8a75b"

REQUIRED_PATHS = (
    "experiment.name",
    "experiment.output_dir",
    "experiment.seed",
    "model.name_or_path",
    "model.trust_remote_code",
    "model.gradient_checkpointing",
    "data.train_file",
    "data.val_file",
    "data.train_batch_size",
    "data.max_prompt_length",
    "data.max_response_length",
    "data.shuffle",
    "data.filter_overlong_prompts",
    "data.truncation",
    "rollout.num_samples_per_prompt",
    "rollout.temperature",
    "rollout.top_p",
    "rollout.top_k",
    "rollout.engine",
    "rollout.mode",
    "rollout.tensor_model_parallel_size",
    "rollout.gpu_memory_utilization",
    "algorithm.name",
    "algorithm.adv_estimator",
    "algorithm.use_kl_in_reward",
    "algorithm.use_kl_loss",
    "algorithm.clip_ratio_low",
    "algorithm.clip_ratio_high",
    "algorithm.loss_aggregation",
    "algorithm.dynamic_sampling.enabled",
    "algorithm.overlong_reward.enabled",
    "optimizer.name",
    "optimizer.learning_rate",
    "optimizer.weight_decay",
    "scheduler.name",
    "scheduler.warmup_steps",
    "training.epochs",
    "training.total_gradient_steps",
    "training.prompt_mini_batch_size",
    "training.micro_batch_size_per_gpu",
    "training.ppo_epochs",
    "distributed.nnodes",
    "distributed.n_gpus_per_node",
    "distributed.strategy",
    "distributed.param_offload",
    "distributed.optimizer_offload",
    "entropy.calculate",
    "entropy.chunked",
    "entropy.chunk_size",
    "entropy.checkpointing",
    "reward.type",
    "reward.implementation",
    "logging.tensorboard.enabled",
    "logging.wandb.enabled",
    "logging.wandb.project",
    "logging.log_interval",
    "checkpoint.save_interval",
    "checkpoint.keep_last",
    "checkpoint.save_hf_model",
    "checkpoint.resume_mode",
    "checkpoint.resume_from_path",
    "evaluation.checkpoint_path",
    "evaluation.benchmark",
    "evaluation.num_samples_per_prompt",
    "evaluation.temperature",
    "evaluation.top_p",
    "evaluation.max_response_length",
    "evaluation.top_k",
    "evaluation.tensor_parallel_size",
    "evaluation.gpu_memory_utilization",
    "evaluation.max_num_seqs",
    "evaluation.during_training_interval",
)


def _get(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing required YAML field: {dotted}")
        value = value[key]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    return config


def validate_config(config: dict[str, Any]) -> None:
    for path in REQUIRED_PATHS:
        _get(config, path)
    method = config["algorithm"]["name"]
    estimator = config["algorithm"]["adv_estimator"]
    if (method, estimator) not in {("grpo", "grpo"), ("rl_zvp", "rl_zvp")}:
        raise ValueError("algorithm.name and adv_estimator must both be grpo or both be rl_zvp")
    if method == "rl_zvp":
        _get(config, "algorithm.rl_zvp.alpha")
        _get(config, "algorithm.rl_zvp.zero_variance_epsilon")
        if not config["entropy"]["calculate"]:
            raise ValueError("RL-ZVP requires entropy.calculate=true")
    if method == "grpo" and config["entropy"]["calculate"]:
        raise ValueError("GRPO controlled baseline requires entropy.calculate=false")
    if config["algorithm"]["use_kl_in_reward"] or config["algorithm"]["use_kl_loss"]:
        raise ValueError("this reproduction requires both KL switches to be false")
    if config["algorithm"]["dynamic_sampling"]["enabled"]:
        raise ValueError("dynamic sampling must be disabled")
    if config["algorithm"]["overlong_reward"]["enabled"]:
        raise ValueError("overlong reward shaping must be disabled")
    if config["rollout"]["num_samples_per_prompt"] < 2:
        raise ValueError("GRPO/RL-ZVP requires at least two responses per prompt")
    if config["data"]["train_batch_size"] % config["training"]["prompt_mini_batch_size"]:
        raise ValueError("train_batch_size must be divisible by prompt_mini_batch_size")
    mini_batches_per_iteration = (
        config["data"]["train_batch_size"]
        // config["training"]["prompt_mini_batch_size"]
        * config["training"]["ppo_epochs"]
    )
    if config["training"]["total_gradient_steps"] % mini_batches_per_iteration:
        raise ValueError("total_gradient_steps must be divisible by optimizer mini-batches per iteration")
    if config["distributed"]["strategy"] not in {"fsdp", "fsdp2"}:
        raise ValueError("distributed.strategy must be fsdp or fsdp2")
    if config["optimizer"]["name"].lower() != "adamw":
        raise ValueError("pinned FSDP recipe currently supports optimizer.name=adamw")
    if config["scheduler"]["name"] not in {"cosine", "constant"}:
        raise ValueError("scheduler.name must be cosine or constant")
    if config["logging"]["log_interval"] != 1:
        raise ValueError("verl v0.9.1 logs each trainer iteration; logging.log_interval must be 1")
    if config["reward"] != {"type": "math", "implementation": "verl"}:
        raise ValueError("this reproduction uses only verl's native MATH reward")
    if config["checkpoint"]["resume_mode"] not in {"auto", "disable", "resume_path"}:
        raise ValueError("checkpoint.resume_mode must be auto, disable, or resume_path")
    if config["checkpoint"]["resume_mode"] == "resume_path" and not config["checkpoint"]["resume_from_path"]:
        raise ValueError("resume_from_path is required when resume_mode=resume_path")


def _resolved_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def prepare_output(config: dict[str, Any]) -> Path:
    output_dir = Path(_resolved_path(config["experiment"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ("logs", "checkpoints", "eval"):
        (output_dir / child).mkdir(exist_ok=True)
    saved = output_dir / "config.yaml"
    if saved.exists():
        previous = load_config(saved)
        if previous != config:
            raise RuntimeError(f"refusing to overwrite different config in {output_dir}; use a new experiment name")
    else:
        with saved.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return output_dir


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _installed_verl_commit() -> str | None:
    """Read the immutable VCS revision recorded by a PEP 610 installation."""
    try:
        distribution = importlib.metadata.distribution("verl")
    except importlib.metadata.PackageNotFoundError:
        return None
    raw = distribution.read_text("direct_url.json")
    if raw:
        try:
            return json.loads(raw).get("vcs_info", {}).get("commit_id")
        except json.JSONDecodeError:
            pass
    try:
        package_root = Path(distribution.locate_file("verl")).resolve().parent
        return subprocess.check_output(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment(config: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch

        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        gpu_models = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        torch_version = torch.__version__
    except ImportError:
        cuda_version, gpu_count, gpu_models, torch_version = None, 0, [], None
    return {
        "verl_version": _package_version("verl"),
        "verl_commit": _installed_verl_commit(),
        "verl_required_commit": EXPECTED_VERL_COMMIT,
        "torch_version": torch_version,
        "transformers_version": _package_version("transformers"),
        "vllm_version": _package_version("vllm"),
        "python_version": platform.python_version(),
        "cuda_version": cuda_version,
        "gpu_models": gpu_models,
        "gpu_count": gpu_count,
        "hostname": socket.gethostname(),
        "model_path": config["model"]["name_or_path"],
        "dataset_paths": [config["data"]["train_file"], config["data"]["val_file"]],
    }


def save_environment(output_dir: Path, config: dict[str, Any]) -> None:
    path = output_dir / "environment.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(collect_environment(config), handle, indent=2, ensure_ascii=False)


def build_verl_config(config: dict[str, Any], output_dir: Path):
    """Compose verl's own config, then map every result-affecting YAML field."""
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf, open_dict

    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        verl_cfg = compose(config_name="ppo_trainer")
    a = config["algorithm"]
    r = config["rollout"]
    d = config["data"]
    t = config["training"]
    dist = config["distributed"]
    entropy = config["entropy"]
    loggers = ["console"]
    if config["logging"]["tensorboard"]["enabled"]:
        loggers.append("tensorboard")
    if config["logging"]["wandb"]["enabled"]:
        loggers.append("wandb")

    with open_dict(verl_cfg):
        verl_cfg.trainer.use_v1 = True
        verl_cfg.trainer.v1.trainer_mode = "sync"
        verl_cfg.data.train_files = [_resolved_path(d["train_file"])]
        verl_cfg.data.val_files = [_resolved_path(d["val_file"])]
        verl_cfg.data.train_batch_size = d["train_batch_size"]
        verl_cfg.data.max_prompt_length = d["max_prompt_length"]
        verl_cfg.data.max_response_length = d["max_response_length"]
        verl_cfg.data.shuffle = d["shuffle"]
        verl_cfg.data.seed = config["experiment"]["seed"]
        verl_cfg.data.filter_overlong_prompts = d["filter_overlong_prompts"]
        verl_cfg.data.truncation = d["truncation"]

        verl_cfg.actor_rollout_ref.model.path = config["model"]["name_or_path"]
        verl_cfg.actor_rollout_ref.model.trust_remote_code = config["model"]["trust_remote_code"]
        verl_cfg.actor_rollout_ref.model.enable_gradient_checkpointing = config["model"]["gradient_checkpointing"]
        verl_cfg.actor_rollout_ref.actor.strategy = dist["strategy"]
        verl_cfg.actor_rollout_ref.actor.fsdp_config.strategy = dist["strategy"]
        verl_cfg.actor_rollout_ref.actor.fsdp_config.param_offload = dist["param_offload"]
        verl_cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload = dist["optimizer_offload"]
        verl_cfg.actor_rollout_ref.actor.ppo_mini_batch_size = t["prompt_mini_batch_size"]
        verl_cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = t["micro_batch_size_per_gpu"]
        verl_cfg.actor_rollout_ref.actor.ppo_epochs = t["ppo_epochs"]
        verl_cfg.actor_rollout_ref.actor.clip_ratio_low = a["clip_ratio_low"]
        verl_cfg.actor_rollout_ref.actor.clip_ratio_high = a["clip_ratio_high"]
        verl_cfg.actor_rollout_ref.actor.loss_agg_mode = a["loss_aggregation"]
        verl_cfg.actor_rollout_ref.actor.use_kl_loss = a["use_kl_loss"]
        verl_cfg.actor_rollout_ref.actor.entropy_coeff = 0.0
        verl_cfg.actor_rollout_ref.actor.calculate_entropy = entropy["calculate"]
        verl_cfg.actor_rollout_ref.actor.entropy_from_logits_with_chunking = entropy["chunked"]
        if entropy["chunk_size"] is not None:
            verl_cfg.actor_rollout_ref.actor.entropy_from_logits_chunk_size = entropy["chunk_size"]
        verl_cfg.actor_rollout_ref.actor.entropy_checkpointing = entropy["checkpointing"]
        verl_cfg.actor_rollout_ref.actor.optim.optimizer = "AdamW"
        verl_cfg.actor_rollout_ref.actor.optim.optimizer_impl = "torch.optim"
        verl_cfg.actor_rollout_ref.actor.optim.lr = config["optimizer"]["learning_rate"]
        verl_cfg.actor_rollout_ref.actor.optim.weight_decay = config["optimizer"]["weight_decay"]
        verl_cfg.actor_rollout_ref.actor.optim.lr_scheduler_type = config["scheduler"]["name"]
        verl_cfg.actor_rollout_ref.actor.optim.lr_warmup_steps = config["scheduler"]["warmup_steps"]
        save_contents = ["model", "optimizer", "extra"]
        if config["checkpoint"]["save_hf_model"]:
            save_contents.append("hf_model")
        verl_cfg.actor_rollout_ref.actor.checkpoint.save_contents = save_contents
        verl_cfg.actor_rollout_ref.actor.checkpoint.load_contents = ["model", "optimizer", "extra"]

        verl_cfg.actor_rollout_ref.rollout.name = r["engine"]
        verl_cfg.actor_rollout_ref.rollout.mode = r["mode"]
        verl_cfg.actor_rollout_ref.rollout.n = r["num_samples_per_prompt"]
        verl_cfg.actor_rollout_ref.rollout.temperature = r["temperature"]
        verl_cfg.actor_rollout_ref.rollout.top_p = r["top_p"]
        verl_cfg.actor_rollout_ref.rollout.top_k = r["top_k"]
        verl_cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = r["tensor_model_parallel_size"]
        verl_cfg.actor_rollout_ref.rollout.gpu_memory_utilization = r["gpu_memory_utilization"]
        verl_cfg.actor_rollout_ref.rollout.seed = config["experiment"]["seed"]

        verl_cfg.algorithm.adv_estimator = a["adv_estimator"]
        verl_cfg.algorithm.use_kl_in_reward = a["use_kl_in_reward"]
        verl_cfg.algorithm.norm_adv_by_std_in_grpo = True
        verl_cfg.algorithm.filter_groups.enable = a["dynamic_sampling"]["enabled"]
        if a["name"] == "rl_zvp":
            verl_cfg.algorithm.rl_zvp = a["rl_zvp"]

        mini_batches_per_iteration = d["train_batch_size"] // t["prompt_mini_batch_size"] * t["ppo_epochs"]
        verl_cfg.trainer.total_epochs = t["epochs"]
        # verl counts rollout/update iterations here; the YAML counts optimizer mini-batches.
        verl_cfg.trainer.total_training_steps = t["total_gradient_steps"] // mini_batches_per_iteration
        verl_cfg.trainer.nnodes = dist["nnodes"]
        verl_cfg.trainer.n_gpus_per_node = dist["n_gpus_per_node"]
        verl_cfg.trainer.project_name = config["logging"]["wandb"]["project"]
        verl_cfg.trainer.experiment_name = config["experiment"]["name"]
        verl_cfg.trainer.logger = loggers
        verl_cfg.trainer.save_freq = config["checkpoint"]["save_interval"]
        verl_cfg.trainer.max_actor_ckpt_to_keep = config["checkpoint"]["keep_last"]
        verl_cfg.trainer.resume_mode = config["checkpoint"]["resume_mode"]
        verl_cfg.trainer.resume_from_path = config["checkpoint"]["resume_from_path"]
        verl_cfg.trainer.test_freq = config["evaluation"]["during_training_interval"]
        verl_cfg.trainer.default_local_dir = str(output_dir / "checkpoints")
        verl_cfg.trainer.validation_data_dir = str(output_dir / "eval")

        runtime_env = verl_cfg.ray_kwargs.ray_init.runtime_env
        runtime_env.working_dir = str(PROJECT_ROOT)
        if runtime_env.get("env_vars") is None:
            runtime_env.env_vars = {}
        runtime_env.env_vars["TENSORBOARD_DIR"] = str(output_dir / "logs")
        if a["name"] == "rl_zvp":
            runtime_env.env_vars["RL_ZVP_INSTALL_BRIDGE"] = "1"
            runtime_env.worker_process_setup_hook = "rl_zvp.verl_bridge"
    OmegaConf.resolve(verl_cfg)
    return verl_cfg


def _check_verl_version() -> None:
    version = _package_version("verl")
    if version is None:
        raise RuntimeError("verl is not installed; install the commit in REQUIRED_VERL.txt")
    normalized = version.removeprefix("v")
    if not normalized.startswith(EXPECTED_VERL_VERSION):
        raise RuntimeError(f"expected verl {EXPECTED_VERL_VERSION}, found {version}")
    commit = _installed_verl_commit()
    if commit is not None and commit != EXPECTED_VERL_COMMIT:
        raise RuntimeError(f"expected verl commit {EXPECTED_VERL_COMMIT}, found {commit}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    output_dir = prepare_output(config)
    save_environment(output_dir, config)
    _check_verl_version()
    os.environ["TENSORBOARD_DIR"] = str(output_dir / "logs")
    verl_config = build_verl_config(config, output_dir)
    from verl.trainer.main_ppo import TaskRunnerV1, run_ppo

    run_ppo(verl_config, task_runner_class=TaskRunnerV1)


if __name__ == "__main__":
    main()
