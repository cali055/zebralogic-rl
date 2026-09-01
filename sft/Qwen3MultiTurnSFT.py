# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


TOOL_DEF = [{
    "type": "function",
    "function": {
        "name": "enter_value",
        "description": "Enter a value into the grid for a specific house and attribute.",
        "parameters": {
            "type": "object",
            "properties": {
                "house_number": {"type": "integer", "description": "The house number (1-indexed)."},
                "attribute":    {"type": "string",  "description": "The attribute name."},
                "value":        {"type": "string",  "description": "The value to assign."}
            },
            "required": ["house_number", "attribute", "value"]
        }
    }
}]


class Qwen3MultiTurnSFTDataset(Dataset):
    """
    Multi-turn SFT dataset for Qwen3-Thinking with tool calls.

    Tokenizes the full conversation once via apply_chat_template, then
    computes per-assistant-turn loss mask spans using prefix-diff logic.
    Loss is applied only to assistant turn tokens (reasoning + tool call),
    not to system/user/tool-response tokens.
    """

    def __init__(self, parquet_files, tokenizer, config, max_samples=-1):
        self.max_length = config.get("max_length", 10240)
        self.truncation = config.get("truncation", "right")
        self.messages_key = config.get("messages_key", "messages")

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]

        dfs = []
        for f in parquet_files:
            local = copy_to_local(f, verbose=True, use_shm=False)
            dfs.append(pd.read_parquet(local))
        self.dataframe = pd.concat(dfs, ignore_index=True)

        if 0 < max_samples < len(self.dataframe):
            self.dataframe = self.dataframe.iloc[:max_samples]

        self._tokenize_all()

    def _tokenize_all(self):
        self.all_input_ids = []
        self.all_attention_mask = []
        self.all_loss_mask = []
        self.all_position_ids = []

        tokenizer = self.tokenizer
        skipped = 0

        for idx in range(len(self.dataframe)):
            row = self.dataframe.iloc[idx]

            # Deserialize messages
            messages = row[self.messages_key]
            if isinstance(messages, str):
                messages = json.loads(messages)
            if hasattr(messages, "tolist"):
                messages = messages.tolist()
            messages = [dict(m) for m in messages]

            # Tokenize full conversation
            full_ids = tokenizer.apply_chat_template(
                messages,
                tools=TOOL_DEF,
                add_generation_prompt=False,
                enable_thinking=True,
                tokenize=True,
            )
            full_ids = torch.tensor(full_ids, dtype=torch.long)
            seq_len = full_ids.shape[0]

            if seq_len > self.max_length:
                if self.truncation == "error":
                    skipped += 1
                    continue
                elif self.truncation == "right":
                    full_ids = full_ids[:self.max_length]
                    seq_len = self.max_length
                elif self.truncation == "left":
                    full_ids = full_ids[-self.max_length:]
                    seq_len = self.max_length

            # Pad to max_length
            input_ids = torch.full((self.max_length,), tokenizer.pad_token_id or 0, dtype=torch.long)
            input_ids[:seq_len] = full_ids

            # Attention mask
            attention_mask = torch.zeros(self.max_length, dtype=torch.int64)
            attention_mask[:seq_len] = 1

            loss_mask = attention_mask.clone()  

            
            position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

            self.all_input_ids.append(input_ids)
            self.all_attention_mask.append(attention_mask)
            self.all_loss_mask.append(loss_mask)
            self.all_position_ids.append(position_ids)

        if skipped > 0:
            print(f"Qwen3MultiTurnSFTDataset: skipped {skipped} examples exceeding max_length={self.max_length}")
        print(f"Qwen3MultiTurnSFTDataset: {len(self.all_input_ids)} examples loaded")

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": self.all_input_ids[item],
            "attention_mask": self.all_attention_mask[item],
            "position_ids": self.all_position_ids[item],
            "loss_mask": self.all_loss_mask[item],
        }