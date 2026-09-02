
import json
import re
from typing import Optional

def extract_solution_json(text: str) -> Optional[dict]:
    after_think = text.split("</think>")[-1]
    match = re.search(r'\{"solution"\s*:\s*\[.*?\]\s*\}', after_think, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def compute_cell_accuracy(predicted: list[dict], ground_truth: list[dict], attributes: list[str]) -> float:
    gt_lookup = {row["house"]: row for row in ground_truth}

    total_cells = len(ground_truth) * len(attributes)
    if total_cells == 0:
        return 0.0

    correct = 0
    for pred_row in predicted:
        house = pred_row.get("house")
        if house not in gt_lookup:
            continue
        gt_row = gt_lookup[house]
        for attr in attributes:
            if pred_row.get(attr) == gt_row.get(attr):
                correct += 1

    return correct / total_cells


def compute_score(data_source: str, solution_str: str, ground_truth: dict, extra_info: dict) -> float:
    gt_solution = ground_truth["solution"]
    attributes = [k for k, v in ground_truth["attributes"].items() if v is not None]
    parsed = extract_solution_json(solution_str)
    if parsed is None:
        return 0.0
    predicted = parsed.get("solution", [])
    if not isinstance(predicted, list) or len(predicted) == 0:
        return 0.0
    cell_acc = compute_cell_accuracy(predicted, gt_solution, attributes)
    puzzle_correct = 1.0 if cell_acc == 1.0 else 0.0
    return cell_acc + puzzle_correct