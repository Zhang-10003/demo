# RL-ZVP reproduction design

## Pinned verl

- Release: `v0.9.1`
- Commit: `1876b06d0a3e4e71e06230be10af14492ca8a75b`
- Source inspected: the tag above, not an older blog/example.
- Trainer: v1 synchronous PPO/GRPO trainer (`trainer.use_v1=true`, `trainer.v1.trainer_mode=sync`).

## Verified training paths

1. Native GRPO estimator: `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`, registered by `register_adv_est(AdvantageEstimator.GRPO)`. It uses outcome reward sum, prompt `uid`, `torch.std` (unbiased for groups larger than one), epsilon `1e-6`, and broadcasts through `response_mask`.
2. Advantage call: `verl/trainer/ppo/v1/trainer_base.py::PPOTrainer._compute_advantage` -> `v1/utils.py::compute_advantage_for_multi_trajectories` -> `ppo/ray_trainer.py::compute_advantage` -> registry estimator for non-native names.
3. Old log probability: `PPOTrainer._compute_old_log_prob` calls `actor_rollout_wg.compute_log_prob` before advantage computation.
4. Token entropy: the same old-log-prob forward sets `calculate_entropy=True`; FSDP computes entropy in `verl/workers/engine/fsdp/transformer_impl.py` using `verl.utils.torch_functional.entropy_from_logits` or its chunked variant.
5. Entropy retention: v1 stores response-aligned `[B,T_response]` entropy in TransferQueue. It is retained, but `_compute_advantage` does not select it into `DataProto` in unmodified v0.9.1.
6. Custom advantage: verl supports string registration through `core_algos.register_adv_est("rl_zvp")`. `rl_zvp.advantage.register_with_verl()` uses that API.
7. Policy loss: `verl/trainer/ppo/core_algos.py::compute_policy_loss_vanilla`, selected by `get_policy_loss_fn`; RL-ZVP does not alter it.
8. Loss aggregation: `core_algos.agg_loss`; YAML `token-mean` maps to `actor_rollout_ref.actor.loss_agg_mode`.

## Minimal bridge

No verl file is copied or edited. `rl_zvp/verl_bridge.py` installs two runtime wrappers only in RL-ZVP Ray workers:

1. During `PPOTrainer._compute_advantage`, add the already stored `entropy` field to the TransferQueue selection.
2. Intercept only estimator name `rl_zvp`, validate/mask/detach entropy, and call the registered RL-ZVP estimator. Every other estimator delegates byte-for-byte to the original verl function.

The hook is loaded with Ray `runtime_env.worker_process_setup_hook`. GRPO does not install the bridge. This is the smallest viable change because v0.9.1's generic custom-estimator call does not forward arbitrary batch fields.

## Data flow

```text
parquet prompt -> verl dataloader -> vLLM rollout (G responses)
-> verl MATH rule reward -> old-log-prob forward (+ token entropy)
-> native GRPO or RL-ZVP advantage -> unchanged verl PPO policy loss
-> FSDP2 backward/AdamW/checkpoint
```

Entropy is an advantage-shaping signal only. It is detached before use; `entropy_coeff=0` and no entropy regularization term is added.

## Batch semantics

- YAML `data.train_batch_size=512` is prompt groups per training iteration.
- YAML `rollout.num_samples_per_prompt=8` expands this to `4096` trajectories.
- YAML `training.prompt_mini_batch_size=32` is prompt groups.
- In v0.9.1's v1 trainer, `actor.ppo_mini_batch_size` is also prompt-group units; train.py maps `32` directly. `PPOTrainer._update_actor` then multiplies it by rollout `n=8`, so the actor receives `256` trajectories per optimizer mini-batch.
- Therefore one iteration has `512 / 32 = 16` optimizer mini-batches.
- The YAML's `672` optimizer gradient steps map to `672 / 16 = 42` verl v1 trainer iterations. Mapping 672 directly to `trainer.total_training_steps` would request 10,752 optimizer mini-batches and is incorrect.
- `micro_batch_size_per_gpu` maps directly to verl's per-GPU trajectory micro-batch.

## YAML to verl mapping

| Project YAML | verl v0.9.1 |
|---|---|
| `model.name_or_path` | `actor_rollout_ref.model.path` |
| `data.train_file/val_file` | `data.train_files/val_files` |
| prompt/response lengths | `data.max_prompt_length/max_response_length` |
| `rollout.*` | `actor_rollout_ref.rollout.*` |
| `algorithm.adv_estimator` | `algorithm.adv_estimator` |
| clip low/high | `actor_rollout_ref.actor.clip_ratio_low/high` |
| loss aggregation | `actor_rollout_ref.actor.loss_agg_mode` |
| LR/weight decay | `actor_rollout_ref.actor.optim.*` |
| scheduler/warmup | `actor_rollout_ref.actor.optim.lr_scheduler_type/lr_warmup_steps` |
| prompt mini-batch | `actor.ppo_mini_batch_size = prompt_mini_batch_size`; v1 expands by `rollout.n` internally |
| GPU/node counts | `trainer.n_gpus_per_node/nnodes` |
| `distributed.strategy` | `actor_rollout_ref.actor.strategy` and FSDP engine strategy |
| checkpoint interval/retention | `trainer.save_freq/max_actor_ckpt_to_keep` |
| gradient steps/epochs | `trainer.total_training_steps = total_gradient_steps / mini_batches_per_iteration`; `trainer.total_epochs` |

Dynamic sampling maps to `algorithm.filter_groups.enable=false`. Overlong reward shaping is not enabled; prompt overflow is an input validation error. KL-in-reward and actor KL loss are both false.
