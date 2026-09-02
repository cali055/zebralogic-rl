#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TRAIN_DATA="${REPO_ROOT}/rl/outcome/FINALGRPOBaseline.parquet"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/rl/outcome}"

export RAY_DISABLE_IMPORT_WARNING=1
export RAY_ADDRESS=local
unset ROCR_VISIBLE_DEVICES
export RAY_DEDUP_LOGS=0

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_DATA}" \
    data.train_batch_size=4 \
    data.max_prompt_length=3500 \
    data.max_response_length=18500 \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.87 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_model_len=40000 \
    custom_reward_function.path="${REPO_ROOT}/rl/outcome/Reward_verl.py" \
    custom_reward_function.name=compute_score \
    ray_kwargs.ray_init.num_cpus=16 \
    trainer.total_training_steps=500 \
    trainer.n_gpus_per_node=1 \
    trainer.project_name=zebralogic_grpo \
    trainer.experiment_name=single_turn_outcome_g16 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${OUTPUT_DIR}"