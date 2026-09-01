#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_DATA="${REPO_ROOT}/sft/SFT_Train_final_split.parquet"
VAL_DATA="${REPO_ROOT}/sft/SFT_Val_final.parquet"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/sft}"

START_TIME=$(date +%s)
echo "Training started at: $(date)"
unset ROCR_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.custom_cls.path="${REPO_ROOT}/sft/Qwen3MultiTurnSFT.py" \
    data.custom_cls.name=Qwen3MultiTurnSFTDataset \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    +data.messages_key=messages \
    data.max_length=18500 \
    data.train_batch_size=24 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain="${MODEL_PATH}"  \
    model.trust_remote_code=True \
    model.lora_rank=32 \
    model.lora_alpha=64 \
    +model.dtype=bfloat16 \
    model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","embed_tokens","lm_head"]' \
    use_remove_padding=False \
    model.enable_gradient_checkpointing=True \
    optim.lr=1e-5 \
    optim.weight_decay=0.1 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.clip_grad=0.5 \
    trainer.total_epochs=10 \
    trainer.project_name=verl-sft-zebra \
    trainer.experiment_name=sft-multiturn-lora \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger='["console","wandb"]'

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "============================================"
echo "Training finished at: $(date)"
echo "Wall clock seconds: ${ELAPSED}"
echo "Wall clock hours: $(echo "scale=2; ${ELAPSED}/3600" | bc)"
echo "============================================"

echo "Multi-turn SFT training finished."