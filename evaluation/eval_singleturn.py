import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional
from enum import Enum

from vllm.lora.request import LoRARequest
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MAX_PROMPT_LENGTH = 4096
MAX_NEW_TOKENS = 32768
LOG_DIR = Path("eval_logs")

SIZES = {
    "small": {"2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3", "4*2"},
    "medium": {"3*4", "3*5", "3*6", "4*3", "4*4", "5*2", "6*2"},
    "large": {"4*5", "5*3", "4*6", "5*4", "6*3"},
    "xlarge": {"5*5", "6*4", "5*6", "6*5", "6*6"},
}


def extract_last_complete_json(text: str) -> Optional[dict]:
    after_think = text.split("</think>")[-1]
    match = re.search(r'\{"solution"\s*:\s*\[.*?\]\s*\}', after_think, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).lower().strip()


def normalize_to_zebralogic_format(
    obj: dict, H: int, active_attrs: list[str]
) -> Optional[dict]:
    if not isinstance(obj, dict) or "solution" not in obj:
        return None

    result = {}
    for row in obj.get("solution", []):
        if not isinstance(row, dict):
            continue
        try:
            house_num = int(row.get("house", -1))
        except (ValueError, TypeError):
            continue
        if house_num < 1 or house_num > H:
            continue

        house_key = f"house {house_num}"
        result[house_key] = {}
        for attr in active_attrs:
            for k, v in row.items():
                if k.lower() == attr.lower() and k.lower() != "house":
                    result[house_key][attr] = v
                    break

    return result if result else None


def convert_gt_to_zebralogic(gt_rows: list[dict], active_attrs: list[str]) -> dict:
    result = {}
    for row in gt_rows:
        if row is None:
            continue
        try:
            house_num = int(row.get("house") or row.get("House"))
        except (ValueError, TypeError):
            continue
        house_key = f"house {house_num}"
        result[house_key] = {
            attr: row[attr] for attr in active_attrs if attr in row
        }
    return result


def compute_cell_accuracy(
    pred: dict, gt: dict, attrs: list[str]
) -> tuple[int, int]:
    correct = total = 0
    for house_key in gt:
        for attr in attrs:
            total += 1
            gt_val = normalize_cell(gt[house_key].get(attr))
            if house_key in pred and attr in pred[house_key]:
                if normalize_cell(pred[house_key].get(attr)) == gt_val:
                    correct += 1
    return correct, total


def build_format_instruction(
    num_houses: int, active_attrs: list[str]
) -> str:
    active_attrs = [a.lower() for a in active_attrs]
    attrs_str = ", ".join(active_attrs)
    example_rows = ", ".join(
        json.dumps({"house": i, **{a: f"{a}_val" for a in active_attrs}})
        for i in range(1, num_houses + 1)
    )
    return (
        f"You are an expert zebra logic grid puzzle solver.\n\n"
        f"Given a set of clues, your task is to determine the unique solution that satisfies all constraints.\n\n"
        f"Output format rules:\n"
        f"- The JSON must have exactly one key \"solution\".\n"
        f"- \"solution\" must contain exactly {num_houses} rows, one per house 1..{num_houses}, "
        f"with all attributes assigned once each.\n"
        f"- Each row must have \"house\" (1 to {num_houses}) and these attributes: {attrs_str}\n"
        f"- No extra text, no code fences, do not include \"reasoning\" inside the JSON.\n\n"
        f"Output exactly:\n{{\"solution\": []}}\n"
        f"Example output:\n{{\"solution\": [{example_rows}]}}"
    )


def score_completion(
    completion: str, gt_rows: list[dict], active_attrs: list[str]
):
    H = len(gt_rows)
    total_cells = H * len(active_attrs)
    gt_table = convert_gt_to_zebralogic(gt_rows, active_attrs)

    obj = extract_last_complete_json(completion)

    failure_reason = None
    if obj is None:
        failure_reason = (
            "json_parse_error"
            if "{" in completion and "}" in completion
            else "no_json_found"
        )

    pred_table = None
    is_valid = False

    if obj is not None:
        pred_table = normalize_to_zebralogic_format(
            obj, H, active_attrs
        )
        if pred_table is None:
            failure_reason = "normalization_failed"
        else:
            expected = {f"house {i}" for i in range(1, H + 1)}
            if set(pred_table.keys()) != expected:
                failure_reason = "missing_houses"
            else:
                for hk in pred_table:
                    if set(pred_table[hk].keys()) != set(active_attrs):
                        failure_reason = "missing_attributes"
                        break

                if failure_reason is None:
                    last_brace = completion.rfind("}")
                    after = (
                        completion[last_brace + 1:].strip()
                        if last_brace != -1
                        else ""
                    )
                    if after and after != "```" and not after.startswith("```"):
                        failure_reason = "extra_content_after_json"
                    else:
                        is_valid = True

    if is_valid:
        correct, total = compute_cell_accuracy(
            pred_table, gt_table, active_attrs
        )
        solved = correct == total
    else:
        correct, total = 0, total_cells
        solved = False

    return {
        "valid_format": is_valid,
        "failure_reason": failure_reason,
        "correct_cells": correct,
        "total_cells": total,
        "solved": solved,
    }


def load_hf_val_dataset(difficulty: str):
    ds = load_dataset(
        "allenai/ZebraLogicBench-private",
        "grid_mode",
        split="test",
    )

    def map_structure(ex):
        header = ex["solution"]["header"]
        rows = ex["solution"]["rows"]
        active_attrs = [
            h.lower() for h in header if h.lower() != "house"
        ]
        house_idx = [
            i for i, h in enumerate(header) if h.lower() == "house"
        ][0]

        solution_rows = []
        for row in rows:
            d = {"house": int(row[house_idx])}
            for h in header:
                if h.lower() != "house":
                    d[h.lower()] = row[header.index(h)]
            solution_rows.append(d)

        return {
            "prompt": ex["puzzle"],
            "target": {"solution": solution_rows},
            "size": ex["size"],
            "active_attrs": active_attrs,
        }

    ds = ds.map(map_structure)
    target = SIZES[difficulty]
    return ds.filter(lambda ex: ex["size"] in target).shuffle(seed=42)


def evaluate(
    llm: LLM,
    tokenizer,
    ds,
    label: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    adapter_path: str = None,
):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    prompts = []
    metadata = []

    for ex in ds:
        gt_rows = ex["target"]["solution"]
        active_attrs = ex["active_attrs"]
        H = len(gt_rows)

        system_prompt = build_format_instruction(H, active_attrs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["prompt"]},
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        token_ids = tokenizer.encode(
            input_text,
            add_special_tokens=False,
        )

        if len(token_ids) > MAX_PROMPT_LENGTH:
            token_ids = token_ids[:MAX_PROMPT_LENGTH]
            input_text = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
            )

        prompts.append(input_text)
        metadata.append({
            "gt_rows": gt_rows,
            "active_attrs": active_attrs,
            "H": H,
        })

    print(f"Generating {len(prompts)} completions with vLLM...")

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    lora_req = (
        LoRARequest("adapter", 1, adapter_path)
        if adapter_path
        else None
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_req,
    )

    total_puzzles = len(outputs)
    total_cells = 0
    correct_cells = 0
    solved_puzzles = 0
    format_errors = 0

    failure_modes = {
        "no_json_found": 0,
        "json_parse_error": 0,
        "normalization_failed": 0,
        "missing_houses": 0,
        "missing_attributes": 0,
        "extra_content_after_json": 0,
    }

    fail_fh = open(
        LOG_DIR / f"{label}_fails_{total_puzzles}.jsonl",
        "w",
        encoding="utf-8",
    )
    ok_fh = open(
        LOG_DIR / f"{label}_oks_{total_puzzles}.jsonl",
        "w",
        encoding="utf-8",
    )

    for i, output in enumerate(outputs):
        completion = output.outputs[0].text
        meta = metadata[i]

        result = score_completion(
            completion,
            meta["gt_rows"],
            meta["active_attrs"],
        )

        total_cells += result["total_cells"]
        correct_cells += result["correct_cells"]

        if result["solved"]:
            solved_puzzles += 1

        if not result["valid_format"]:
            format_errors += 1
            if result["failure_reason"] in failure_modes:
                failure_modes[result["failure_reason"]] += 1

        log_entry = {
            "id": i + 1,
            "valid_format": result["valid_format"],
            "failure_reason": result["failure_reason"],
            "correct_cells": result["correct_cells"],
            "total_cells_expected": result["total_cells"],
            "solved": result["solved"],
            "completion": completion,
        }

        fh = ok_fh if result["solved"] else fail_fh
        fh.write(
            json.dumps(log_entry, ensure_ascii=False) + "\n"
        )
        fh.flush()

    fail_fh.close()
    ok_fh.close()

    cell_acc = (
        correct_cells / total_cells * 100
        if total_cells > 0
        else 0.0
    )
    puzzle_acc = (
        solved_puzzles / total_puzzles * 100
        if total_puzzles > 0
        else 0.0
    )
    format_err_rate = (
        format_errors / total_puzzles * 100
        if total_puzzles > 0
        else 0.0
    )

    print(f"\n{'='*40}")
    print(f"RESULTS: {label}")
    print(f"{'='*40}")
    print(f"Total Puzzles:      {total_puzzles}")
    print(
        f"1. Cell Accuracy:   "
        f"{cell_acc:.2f}% ({correct_cells}/{total_cells})"
    )
    print(
        f"2. Puzzle Accuracy: "
        f"{puzzle_acc:.2f}% ({solved_puzzles}/{total_puzzles})"
    )
    print(
        f"3. Format Error:    "
        f"{format_err_rate:.2f}% ({format_errors}/{total_puzzles})"
    )
    print(f"\nFORMAT FAILURE BREAKDOWN:")

    for k, v in failure_modes.items():
        print(f"  - {k}: {v}")

    print(f"{'='*40}")

    return {
        "label": label,
        "cell_acc": cell_acc,
        "puzzle_acc": puzzle_acc,
        "format_err_rate": format_err_rate,
        "total": total_puzzles,
        "failure_modes": failure_modes,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="HF model ID or local merged model path",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter path (optional, merge first is simpler)",
    )
    parser.add_argument(
        "--difficulty",
        nargs="+",
        default=["small", "medium", "large", "xlarge"],
        choices=["small", "medium", "large", "xlarge"],
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--gpu_memory",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label prefix for logs",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling parameter",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=MAX_PROMPT_LENGTH + args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    if args.adapter:
        llm_kwargs["enable_lora"] = True

    print(f"Loading model: {args.model}")
    llm = LLM(**llm_kwargs)

    label_prefix = args.label or Path(args.model).name

    all_results = []

    for difficulty in args.difficulty:
        print(f"\n{'#'*60}")
        print(f"  DIFFICULTY: {difficulty}")
        print(f"{'#'*60}")

        ds = load_hf_val_dataset(difficulty)
        print(f"Evaluating {len(ds)} puzzles")

        label = f"{label_prefix}_{difficulty}"

        result = evaluate(
            llm,
            tokenizer,
            ds,
            label,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            adapter_path=args.adapter,
        )

        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(
        f"{'Difficulty':<20} "
        f"{'Puzzle Acc':>12} "
        f"{'Cell Acc':>12} "
        f"{'Format Err':>12}"
    )
    print(f"{'-'*60}")

    for r in all_results:
        print(
            f"{r['label']:<20} "
            f"{r['puzzle_acc']:>11.2f}% "
            f"{r['cell_acc']:>11.2f}% "
            f"{r['format_err_rate']:>11.2f}%"
        )

    print(f"{'='*60}")


if __name__ == "__main__":
    main()