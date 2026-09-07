#!/bin/bash

BASE_MODEL=${BASE_MODEL:-/path/to/sft-base-model}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/path/to/grpo-checkpoint}
TARGET_BASE=${TARGET_BASE:-/path/to/merged-output}

STEPS=${STEPS:-500}

for step in $STEPS; do
    echo "=== Step $step ==="

    ADAPTER_DIR="${TARGET_BASE}/step${step}/lora_adapter"
    MERGE_DIR="${TARGET_BASE}/step${step}-fullmerge"

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${CHECKPOINT_DIR}/global_step_${step}/actor" \
        --target_dir "${TARGET_BASE}/step${step}" \
        --trust-remote-code

    sed -i 's/"lora_alpha": 0/"lora_alpha": 64/' \
        "${ADAPTER_DIR}/adapter_config.json"

    sed -i 's|"base_model_name_or_path": null|"base_model_name_or_path": "'"${BASE_MODEL}"'"|' \
        "${ADAPTER_DIR}/adapter_config.json"

    sed -i 's/"inference_mode": false/"inference_mode": true/' \
        "${ADAPTER_DIR}/adapter_config.json"

    echo "adapter_config.json after patching:"
    cat "${ADAPTER_DIR}/adapter_config.json"

    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    '${BASE_MODEL}',
    torch_dtype=torch.bfloat16,
    device_map='cpu',
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base,
    '${ADAPTER_DIR}'
)

merged = model.merge_and_unload()
merged.save_pretrained('${MERGE_DIR}')

tok = AutoTokenizer.from_pretrained(
    '${BASE_MODEL}',
    trust_remote_code=True
)

tok.save_pretrained('${MERGE_DIR}')

print('Done step ${step}')
"

    echo "=== Merge done for step $step ==="
done