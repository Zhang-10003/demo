# RL-ZVP Reproduction on verl

Controlled comparison of:

1. Qwen3-1.7B-Base + MATH + native verl GRPO
2. Qwen3-1.7B-Base + MATH + RL-ZVP

Training:

```bash
python train.py --config configs/grpo_qwen3_1.7b_math.yaml
python train.py --config configs/rl_zvp_qwen3_1.7b_math.yaml
```

Evaluation:

```bash
python evaluate.py --config configs/grpo_qwen3_1.7b_math.yaml
```

Unit tests:

```bash
pytest tests/
```

## Algorithm

For mixed-reward prompt groups, RL-ZVP is exactly pinned verl GRPO:

`A_i = (R_i - mean(R_g)) / (std(R_g) + epsilon)`, broadcast to response tokens.

For an all-correct group:

`A_i,t = alpha * H_i,t`.

For an all-wrong group:

`A_i,t = -alpha * (max_valid_token(H_i) - H_i,t)`.

The maximum is per response, never across responses, prompts, batches, or padding. Entropy is detached and is only advantage shaping. The unchanged verl PPO objective receives the resulting advantage; `entropy_coeff=0`.

## Layout and entrypoints

- `train.py`: only training entrypoint; validates/saves YAML, maps it to verl, installs RL-ZVP only when selected, and calls the verl trainer.
- `evaluate.py`: only benchmark evaluation entrypoint; samples MATH500 with vLLM and writes model-performance metrics.
- `tests/`: tensor/config correctness tests, not benchmark evaluation.
- `rl_zvp/advantage.py`: the complete new algorithm.
- `rl_zvp/verl_bridge.py`: passes entropy already computed by verl to the estimator.
- `data/prepare_math.py`: MATH/MATH500 parquet preparation.
- `DESIGN.md`: inspected verl call paths and exact mapping.

## Pinned environment

The only supported verl baseline is:

```text
v0.9.1
1876b06d0a3e4e71e06230be10af14492ca8a75b
```

See `REQUIRED_VERL.txt`. Use Linux with NVIDIA GPUs. verl v0.9.1 pins CUDA-specific Torch/vLLM builds; its official `uv` environment is preferred. A direct installation is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The same commit must be used for GRPO and RL-ZVP. `train.py` rejects another verl release. v0.9.1's v1 trainer already computes token entropy in the old-log-prob forward, so RL-ZVP does not add a second model forward.

## Base model

Keep initial weights outside this repository, for example:

```text
/path/to/models/Qwen3-1.7B-Base
```

Set the identical `model.name_or_path` in both formal YAML files. A fixed local snapshot is recommended; `Qwen/Qwen3-1.7B-Base` is also accepted. Both runs must start from the same immutable checkpoint.

## Data

Prepare both datasets with the shared YAML prompt:

```bash
python data/prepare_math.py --config configs/grpo_qwen3_1.7b_math.yaml
```

Outputs:

```text
data/processed/math_train.parquet
data/processed/math500.parquet
```

The converter follows verl's official MATH schema and uses verl's official boxed-answer extraction/reward route. The exact original RL-ZVP prompt template was not available; this reproduction fixes the following template in YAML for both methods:

```text
Let's think step by step and output the final answer within \boxed{}.
```

Raw sources are `DigitalLearningGmbH/MATH-lighteval` and `HuggingFaceH4/MATH-500`.

## YAML-only configuration

YAML is the experiment source of truth. Model/data paths, rollout budget, generation, optimizer, scheduler, clipping, KL switches, entropy, hardware, logging, checkpointing, training and evaluation are explicit. Shell scripts select only a YAML file. Change GPU count only in YAML.

The formal configurations are intentionally identical except experiment identity, estimator identity, RL-ZVP parameters, and `entropy.calculate`. `tests/test_config.py` enforces this controlled comparison.

## Batch-size mapping

The paper setting is:

```text
512 prompt groups/iteration * 8 rollouts = 4096 trajectories/iteration
32 prompt groups/mini-batch * 8 rollouts = 256 trajectories/mini-batch
512 / 32 = 16 optimizer mini-batches/iteration
672 optimizer gradient steps / 16 = 42 verl trainer iterations
```

In pinned verl v0.9.1's v1 trainer, `actor.ppo_mini_batch_size` is prompt-group units. `train.py` therefore maps YAML `prompt_mini_batch_size: 32` directly to `ppo_mini_batch_size=32`; `PPOTrainer._update_actor` multiplies it by rollout `n=8` before actor training, producing the required 256 trajectory mini-batch. verl's `trainer.total_training_steps` counts rollout/update iterations, so `train.py` maps the paper's 672 optimizer mini-batches to 42 trainer iterations. This differs from older verl interfaces/examples where users often supplied a sequence-level value directly.

## Debug workflow

```bash
python train.py --config configs/debug_grpo.yaml
python train.py --config configs/debug_rl_zvp.yaml
```

The first validates the native GRPO path. The second additionally validates entropy -> ZVP classification -> token advantage. Debug results are NOT comparable to paper results.

Formal runs:

```bash
python train.py --config configs/grpo_qwen3_1.7b_math.yaml
python train.py --config configs/rl_zvp_qwen3_1.7b_math.yaml
```

Resolved config and environment metadata are written to `outputs/<experiment>/config.yaml` and `environment.json`. A different existing config is never silently overwritten; the explicit default `checkpoint.resume_mode: auto` resumes the latest checkpoint only when the saved config is identical. Checkpoints are under `outputs/<experiment>/checkpoints/global_step_*`; the last training step is always saved by verl. `checkpoint.save_hf_model: true` also writes directly loadable Hugging Face weights.

TensorBoard:

```bash
tensorboard --logdir outputs
```

WandB is optional and disabled by default.

## MATH500 evaluation

Set, in the relevant YAML:

```yaml
evaluation:
  checkpoint_path: outputs/grpo_qwen3_1.7b_math/checkpoints/final
```

`final` is a virtual alias resolved by `evaluate.py` to the highest `global_step_*`; an explicit HF checkpoint directory is also accepted. Then run:

```bash
python evaluate.py --config configs/grpo_qwen3_1.7b_math.yaml
python evaluate.py --config configs/rl_zvp_qwen3_1.7b_math.yaml
```

Results go to `outputs/<experiment>/eval/results.json`.

- `Acc@8`: mean correctness over all 8 sampled responses for all questions.
- `Pass@8`: fraction of questions with at least one correct response among 8.
- `avg_response_length`: mean generated token count.

Evaluation uses the same verl MATH verifier as training.

## Reproduction differences and constraints

- Original exact prompt wording was unavailable; the fixed shared prompt above is disclosed.
- The implementation targets only the pinned verl commit and v1 sync trainer.
- verl's native unbiased `torch.std` GRPO normalization is preserved.
- Dynamic sampling, zero-variance filtering, overlong reward shaping, KL reward and KL loss are disabled.
- FSDP/FSDP2, Ray, vLLM, optimization, policy loss, checkpointing and distributed execution remain owned by verl.

## Adding another advantage estimator

Add one small function with verl's `(advantages, returns)` contract, register its string through `register_adv_est`, add only the required batch-field bridge if verl does not already forward that signal, and add a YAML plus tensor/config tests. Do not create another trainer loop.
