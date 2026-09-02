import json
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

TOOL_SCHEMA = OpenAIFunctionToolSchema(**{
    "type": "function",
    "function": {
        "name": "enter_value",
        "description": "Updates a single grid cell with a value deduced from puzzle clues.",
        "parameters": {
            "type": "object",
            "properties": {
                "house_number": {
                    "type": "integer",
                    "description": "House number (1-indexed)",
                    "minimum": 1,
                },
                "attribute": {
                    "type": "string",
                    "description": "Attribute name"
                },
                "value": {
                    "type": "string",
                    "description": "Value to assign"
                }
            },
            "additionalProperties": False,
            "required": ["house_number", "attribute", "value"]
        }
    }
})

_GRID_KEY = "enter_value_grid"
_GT_KEY = "enter_value_gt"


def _format_zebra_grid(grid: dict, houses: list[int], attributes: list[str], all_values: list[str] = None) -> str:
    value_lens = [len(v) for v in all_values] if all_values else []
    col_w = max([len("house")] + [len(a) for a in attributes] + value_lens) + 2
    header = "house".ljust(col_w) + "".join(a.ljust(col_w) for a in attributes)
    sep = "-" * len(header)
    rows = [header, sep]
    for h in houses:
        cells = grid.get(h, {})
        row = str(h).ljust(col_w)
        for a in attributes:
            row += str(cells.get(a, ".")).ljust(col_w)
        rows.append(row)
    return "\n" + "\n".join(rows) + "\n"


class EnterValueTool(BaseTool):

    def __init__(self, config: dict, tool_schema=None):
        super().__init__(config, tool_schema or TOOL_SCHEMA)

    async def create(self, instance_id: Optional[str] = None, **kwargs):
        import uuid
        if instance_id is None:
            instance_id = str(uuid.uuid4())

        self._pending_create_kwargs = kwargs.get("create_kwargs", {})

        return instance_id, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")

        if agent_data is not None and _GRID_KEY not in agent_data.extra_fields:
            agent_data.extra_fields[_GRID_KEY] = {}

        if agent_data is not None and _GT_KEY not in agent_data.extra_fields:
            ck = getattr(self, "_pending_create_kwargs", {})
            gt_grid = ck.get("gt_grid", {})
            active_attrs = list(ck.get("active_attrs", []))
            # derive houses from non-None gt_grid keys
            houses = sorted([int(k) for k, v in gt_grid.items() if v is not None])
            # derive all_values from gt_grid active attrs only, excluding None
            all_values = list(set(
                v for h_data in gt_grid.values()
                if h_data is not None
                for a, v in h_data.items()
                if a in active_attrs and v is not None
            ))
            agent_data.extra_fields[_GT_KEY] = {
                "gt_grid": gt_grid,
                "n_cells": ck.get("n_cells", 0),
                "active_attrs": active_attrs,
                "houses": houses,
                "all_values": all_values,
                "correct_cells": [],
                "filled_cells": [],
                "puzzle_solved": False,
            }

        if agent_data is not None and "tool_cells" not in agent_data.extra_fields:
            agent_data.extra_fields["tool_cells"] = []

        house_number = parameters.get("house_number")
        attribute = parameters.get("attribute")
        value = parameters.get("value")

        r_cell = 0.0
        malformed = False

        if agent_data is not None and house_number is not None and attribute is not None and value is not None:
            grid = agent_data.extra_fields[_GRID_KEY]
            house_str = str(house_number)

            gt_info = agent_data.extra_fields.get(_GT_KEY)
            if gt_info and gt_info["n_cells"] > 0:
                gt_grid = gt_info["gt_grid"]
                active_attrs = gt_info["active_attrs"]
                houses = gt_info["houses"]
                cell_key = [house_str, attribute]

                bad_house = house_str not in gt_grid or gt_grid[house_str] is None
                bad_attribute = (not bad_house) and (attribute not in active_attrs)

                if bad_house:
                    agent_data.extra_fields["tool_cells"].append(("INVALID", "INVALID"))
                    r_cell = -2.5
                    malformed = True
                    valid_houses = ", ".join(str(h) for h in houses)
                    response_text = (
                        f"Error: invalid house_number '{house_number}'. "
                        f"Valid house numbers are: {valid_houses}"
                    )
                elif bad_attribute:
                    agent_data.extra_fields["tool_cells"].append(("INVALID", "INVALID"))
                    r_cell = -2.5
                    malformed = True
                    valid_attrs = ", ".join(active_attrs)
                    response_text = (
                        f"Error: invalid attribute '{attribute}'. "
                        f"Valid attributes are: {valid_attrs}"
                    )
                else:
                    is_correct = (gt_grid[house_str].get(attribute) == value)
                    is_first = cell_key not in gt_info["filled_cells"]
                    prev_correct = (cell_key in gt_info["correct_cells"])

                    # six-way transition reward
                    if is_first:
                        r_cell = 1.0 if is_correct else -1.0
                    else:
                        if prev_correct and is_correct:
                            r_cell = -1.5   # correct -> correct (redundant re-entry)
                        elif prev_correct and not is_correct:
                            r_cell = -1.5   # correct -> incorrect
                        elif (not prev_correct) and is_correct:
                            r_cell = 0.5    # incorrect -> correct
                        else:
                            r_cell = -1.0   # incorrect -> incorrect

                    if is_first:
                        gt_info["filled_cells"].append(cell_key)

                    if is_correct and cell_key not in gt_info["correct_cells"]:
                        gt_info["correct_cells"].append(cell_key)
                    elif not is_correct and cell_key in gt_info["correct_cells"]:
                        gt_info["correct_cells"].remove(cell_key)

                    n = gt_info["n_cells"]
                    gt_info["puzzle_solved"] = (
                        len(gt_info["correct_cells"]) == n and len(gt_info["filled_cells"]) == n
                    )

                    agent_data.extra_fields["tool_cells"].append((house_str, attribute))
            else:
                # No GT info available (n_cells == 0)
                agent_data.extra_fields["tool_cells"].append(("INVALID", "INVALID"))
                r_cell = -2.5
                malformed = True
                response_text = "Error: invalid house or attribute"

            # update grid state (int key); skip on malformed to avoid poisoning
            if not malformed:
                if house_number not in grid:
                    grid[house_number] = {}
                grid[house_number][attribute] = value

                gt_info = agent_data.extra_fields.get(_GT_KEY, {})
                houses = gt_info.get("houses", [house_number])
                active_attrs = gt_info.get("active_attrs", [attribute])
                all_values = gt_info.get("all_values", [])

                response_text = (
                    f"Value '{value}' placed at (house {house_number}, {attribute}). "
                    f"Current Board Status:{_format_zebra_grid(grid, houses, active_attrs, all_values)}"
                )
        else:
            if agent_data is not None:
                agent_data.extra_fields["tool_cells"].append(("INVALID", "INVALID"))
            r_cell = -2.5
            response_text = "Error: missing required parameters"

        return ToolResponse(text=response_text), r_cell, {}

    async def release(self, instance_id: str, **kwargs):
        pass