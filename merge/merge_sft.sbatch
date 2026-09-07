#!/bin/bash
BASE_MODEL=${BASE_MODEL:-/path/to/base-model}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/path/to/sft-checkpoint}
TARGET_BASE=${TARGET_BASE:-/path/to/merged-output}

STEP=${STEP:-100}

echo "=== Step ${STEP} ==="

ADAPTER_DIR="${TARGET_BASE}/step${STEP}/lora_adapter"
MERGE_DIR="${TARGET_BASE}/step${STEP}-fullmerge"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CHECKPOINT_DIR}/global_step_${STEP}" \
    --target_dir "${TARGET_BASE}/step${STEP}" \
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
from safetensors.torch import load_file
import torch, json

base = AutoModelForCausalLM.from_pretrained(
    '${BASE_MODEL}',
    torch_dtype=torch.bfloat16,
    device_map='cpu',
    trust_remote_code=True,
    tie_word_embeddings=False
)

base.lm_head.weight.data = base.model.embed_tokens.weight.data.clone()

embed_before = base.model.embed_tokens.weight.data.clone()
lm_head_before = base.lm_head.weight.data.clone()

model = PeftModel.from_pretrained(base, '${ADAPTER_DIR}')
merged = model.merge_and_unload()

# --- Manual merge for embed_tokens and lm_head (PEFT's automatic merge is unreliable for tied-then-untied layers, see huggingface/peft#2018) ---
adapter_weights = load_file('${ADAPTER_DIR}/adapter_model.safetensors')

with open('${ADAPTER_DIR}/adapter_config.json') as f:
    cfg = json.load(f)

alpha = cfg['lora_alpha']
r = cfg['r']
scaling = alpha / r

A_emb = adapter_weights[
    'base_model.model.model.embed_tokens.lora_embedding_A.default'
].float()

B_emb = adapter_weights[
    'base_model.model.model.embed_tokens.lora_embedding_B.default'
].float()

delta_emb = (B_emb @ A_emb).T * scaling
new_embed = (
    embed_before.float() + delta_emb
).to(merged.model.embed_tokens.weight.dtype)

A_lm = adapter_weights[
    'base_model.model.lm_head.lora_A.weight'
].float()

B_lm = adapter_weights[
    'base_model.model.lm_head.lora_B.weight'
].float()

delta_lm = (B_lm @ A_lm) * scaling
new_lm_head = (
    lm_head_before.float() + delta_lm
).to(merged.lm_head.weight.dtype)

print('embed delta abs mean (manual):', delta_emb.abs().mean().item())
print('lm_head delta abs mean (manual):', delta_lm.abs().mean().item())

merged.model.embed_tokens.weight.data = new_embed
merged.lm_head.weight.data = new_lm_head

merged.config.tie_word_embeddings = False

merged.save_pretrained('${MERGE_DIR}')

tok = AutoTokenizer.from_pretrained(
    '${BASE_MODEL}',
    trust_remote_code=True
)

tok.save_pretrained('${MERGE_DIR}')

print('Done step ${STEP}')
"

echo "=== Merge done ==="