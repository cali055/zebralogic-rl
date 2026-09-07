#!/bin/bash
EVAL_PARQUET=${EVAL_PARQUET:-/path/to/eval_zebralogic.parquet}
BASE_CKPT_DIR=${BASE_CKPT_DIR:-/path/to/grpo-checkpoint}
BASE_DUMP_DIR=${BASE_DUMP_DIR:-/path/to/eval-output}
TOOL_CONFIG=${TOOL_CONFIG:-/path/to/ToolConfig.yaml}

STEPS=${STEPS:-500}

for STEP in $STEPS; do
    CHECKPOINT=$BASE_CKPT_DIR/step${STEP}-fullmerge
    DUMP_DIR=$BASE_DUMP_DIR/step${STEP}
    EXPERIMENT_NAME=step${STEP}

    echo "=== Evaluating step ${STEP} ==="
    mkdir -p $DUMP_DIR

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo_process \
        algorithm.use_kl_in_reward=False \
        data.train_files=$EVAL_PARQUET \
        data.val_files=$EVAL_PARQUET \
        data.train_batch_size=4 \
        data.max_prompt_length=5000 \
        data.max_response_length=32768 \
        data.filter_overlong_prompts=True \
        data.return_raw_chat=True \
        data.shuffle=False \
        +data.need_tools_kwargs=True \
        actor_rollout_ref.model.path=$CHECKPOINT \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=3 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.rollout.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.load_format=auto \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.max_model_len=32768 \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=70 \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=3500 \
        reward_model.use_reward_loop=False \
        "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=process_agent_loop.ProcessAgentLoopManager" \
        +ray_kwargs.ray_init.include_dashboard=False \
        trainer.total_training_steps=0 \
        trainer.n_gpus_per_node=1 \
        trainer.project_name=zebralogic_eval \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=0 \
        trainer.val_before_train=True \
        trainer.validation_data_dir=$DUMP_DIR \
        trainer.default_hdfs_dir=null \
        ray_kwargs.ray_init.num_cpus=16

    echo "=== step ${STEP} done ==="
done

echo ""
echo "=== all evals done ==="