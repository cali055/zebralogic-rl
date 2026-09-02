#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/agent:${PYTHONPATH}"

TRAIN_DATA="${REPO_ROOT}/rl/dataset/FINALGRPO.parquet"
MERGED_MODEL="${MODEL_PATH:?Set MODEL_PATH to the merged SFT checkpoint}"
TOOL_CONFIG="${REPO_ROOT}/agent/ToolConfig.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/rl/process/curriculum}"

export HYDRA_FULL_ERROR=1
export RAY_DISABLE_IMPORT_WARNING=1
unset RAY_ADDRESS
export RAY_ADDRESS=local
export RAY_DEDUP_LOGS=0
unset ROCR_VISIBLE_DEVICES
ulimit -Sn 65535
unset TORCH_DISTRIBUTED_DEBUG
unset NCCL_SOCKET_IFNAME
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_process_std \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_DATA}" \
    data.train_batch_size=4 \
    data.max_prompt_length=2500 \
    data.max_response_length=18500 \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    data.shuffle=False \
    +data.need_tools_kwargs=True \
    actor_rollout_ref.model.path="${MERGED_MODEL}" \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.87 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.max_model_len=40000 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=52 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    reward_model.use_reward_loop=False \
    "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=process_agent_loop.ProcessAgentLoopManager" \
    +ray_kwargs.ray_init.include_dashboard=False \
    trainer.total_training_steps=500 \
    trainer.n_gpus_per_node=1 \
    trainer.project_name=zebralogic_grpo_process \
    trainer.experiment_name=process_reward_curriculum \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    ray_kwargs.ray_init.num_cpus=16