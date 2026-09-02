"""
Flow recap
----------
tool_agent_loop.py  ->  AgentLoopOutput.extra_fields["tool_rewards"] = [r0, r1, ..., rK-1]
                         AgentLoopOutput.extra_fields["tool_cells"] = [(h,a), ...]
                         AgentLoopOutput.extra_fields["puzzle_solved"] = True/False

_agent_loop_postprocess  ->  _InternalAgentLoopOutput.extra_fields preserved

_postprocess        ->    reads tool_rewards + response_mask per sample
                          finds 1->0 transitions in response_mask (end of each assistant turn)
                          applies beta * r_puzzle to non-zero tool_rewards (r_puzzle from puzzle_solved)
                          applies flat format_bonus (+0.5) to every valid tool call
                          places final reward at turn-end position k in rm_scores
                          stores tool_cells in output.non_tensor_batch for core_algos.py
                          sets batch["rm_scores"]

ray_trainer.py  ->  _compute_or_extract_reward sees rm_scores, extracts directly
                     token_level_scores = token_level_rewards = rm_scores
                     compute_advantage(adv_estimator="grpo_process") uses our sparse tensor
                     passes non_tensor_batch["tool_cells"] to core_algos.py
"""

import logging
import math
import os
from typing import Any
import sys
import ray
import torch
from omegaconf import DictConfig
from verl.utils.profiler import simple_timer
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorkerBase,
    _InternalAgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.protocol import DataProto

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("tool_agent")
class ProcessToolAgentLoop(ToolAgentLoop):
    """ToolAgentLoop with per-sample max_assistant_turns cap: ceil(M * N * 1.5)."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Any:
        extra_info = kwargs.get("extra_info", {})
        if os.getenv("VERL_IS_EVAL", "0") != "1":
            M = extra_info.get("M", 6)
            N = extra_info.get("N", 6)
            per_sample_max = min(M * N + 10, math.ceil(M * N * 1.5))
            tools_kwargs = dict(kwargs.get("tools_kwargs", {}))
            tools_kwargs["max_assistant_turns"] = per_sample_max
            kwargs["tools_kwargs"] = tools_kwargs
        else:
            logger.warning("eval mode")
            
        output = await super().run(sampling_params, **kwargs)
        output.extra_fields["extra_info"] = extra_info
        if os.getenv("VERL_IS_EVAL", "0") == "1":
            logging.basicConfig(force=True)
            logger.warning(f"[RUN] done M={extra_info.get('M')} N={extra_info.get('N')} solved={output.extra_fields.get('puzzle_solved')}")
            sys.stderr.flush()
        return output

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        per_sample_max = agent_data.tools_kwargs.get("max_assistant_turns", self.max_assistant_turns)
        if per_sample_max and agent_data.assistant_turns >= per_sample_max:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        try:
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        except Exception as e:
            logger.warning(f"[ProcessToolAgentLoop] Failed to decode tool call: {e}")
            agent_data.tool_calls = []
            # record invalid call for reward
            agent_data.tool_rewards.append(-2.5)
            if "tool_cells" not in agent_data.extra_fields:
                agent_data.extra_fields["tool_cells"] = []
            agent_data.extra_fields["tool_cells"].append(("INVALID", "INVALID"))
            logger.warning(f"[ProcessToolAgentLoop] INVALID appended, tool_cells now: {agent_data.extra_fields['tool_cells']}")
            # inject error message and retry
            assistant_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            agent_data.messages.append({"role": "assistant", "content": assistant_text})
            agent_data.messages.append({"role": "tool", "content": "Error: malformed tool call. Please retry with valid JSON.", "name": "enter_value"})
            # encode error message into prompt_ids and response_mask
            error_ids = await self.apply_chat_template(
                [{"role": "tool", "content": "Error: malformed tool call. Please retry with valid JSON.", "name": "enter_value"}],
                remove_system_prompt=True,
            )
            agent_data.prompt_ids += error_ids
            agent_data.response_mask += [0] * len(error_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(error_ids)
            agent_data.user_turns += 1
            return AgentState.GENERATING

        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED



BETA = 0.5
FORMAT_BONUS = 0.5
PENALTY = -1.1

class ProcessAgentLoopWorkerBase(AgentLoopWorkerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_counter = 0
        #self._eval_dump_path = os.getenv("VERL_EVAL_DUMP_PATH", None)
        self._is_eval = os.getenv("VERL_IS_EVAL", "0") == "1"

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        output: DataProto = super()._postprocess(inputs)

        bs = len(inputs)
        response_length = inputs[0].response_mask.shape[-1]
        rm_scores = torch.zeros(bs, response_length, dtype=torch.float32)

        # collect tool_cells per rollout for core_algos.py
        batch_tool_cells = []

        for i, inp in enumerate(inputs):
            tool_rewards = inp.extra_fields.get("tool_rewards", [])
            tool_cells = inp.extra_fields.get("tool_cells", [])
            while len(tool_cells) < len(tool_rewards):
                tool_cells.append(None)
            batch_tool_cells.append(tool_cells)

            if not tool_rewards:
                mask = inp.response_mask.squeeze(0)
                last_pos = (mask.long() == 1).nonzero(as_tuple=False)
                if len(last_pos) > 0:
                    n_cells = inp.extra_fields.get("extra_info", {}).get("tools_kwargs", {}).get("enter_value", {}).get("create_kwargs", {}).get("n_cells", 0)
                    rm_scores[i, last_pos[-1].item()] = PENALTY * n_cells  
                logger.warning(f"[DEBUG i={i}] no tool_rewards straight JSON penalty {PENALTY * n_cells}")
                continue

            # find turn-end positions (1->0 transitions in response_mask)
            mask = inp.response_mask.squeeze(0)
            mask_int = mask.long()
            padded = torch.cat([mask_int, mask_int.new_zeros(1)])
            is_transition = (padded[:-1] == 1) & (padded[1:] == 0)
            turn_ends = is_transition.nonzero(as_tuple=False).squeeze(-1)
            if turn_ends.dim() == 0:
                turn_ends = turn_ends.unsqueeze(0)
            n_transitions = len(turn_ends)

            if n_transitions == 0:
                continue

            # read puzzle_solved directly from enter_value_tool via extra_fields
            puzzle_solved = inp.extra_fields.get("puzzle_solved", False)
            r_puzzle = BETA * (1.0 if puzzle_solved else -1.0)
            n_cells = inp.extra_fields.get("extra_info", {}).get("tools_kwargs", {}).get("enter_value", {}).get("create_kwargs", {}).get("n_cells", 0)

            # assign rewards to tool call turns
            n_place = min(n_transitions, len(tool_rewards))
            for k in range(n_place):
                pos = turn_ends[k].item()
                r_cell = float(tool_rewards[k])
                is_valid = (tool_cells[k] is not None) and (tool_cells[k] != ("INVALID", "INVALID"))
                format_bonus = FORMAT_BONUS if (is_valid and k < n_cells) else 0.0
                r_t = r_cell + r_puzzle + format_bonus
                rm_scores[i, pos] = r_t

            logger.warning(
                f"[DEBUG i={i}] tool_rewards={tool_rewards} tool_cells={tool_cells} "
                f"n_transitions={n_transitions} puzzle_solved={puzzle_solved} "
                f"n_cells={n_cells} r_puzzle={r_puzzle} "
                f"nonzero_rm={[(p.item(), round(rm_scores[i, p].item(), 3)) for p in rm_scores[i].nonzero(as_tuple=False).squeeze(-1)]}"
            )

        output.batch["rm_scores"] = rm_scores
        import numpy as np
        # store tool_cells for cell-identity advantage normalization in core_algos.py
        tool_cells_arr = np.empty(len(batch_tool_cells), dtype=object)
        for j, tc in enumerate(batch_tool_cells):
            tool_cells_arr[j] = tc
        output.non_tensor_batch["tool_cells"] = tool_cells_arr

        
        return output


@ray.remote
class ProcessAgentLoopWorker(ProcessAgentLoopWorkerBase):
    """Ray remote actor: ProcessAgentLoopWorkerBase with Ray decoration."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        super().__init__(config, server_handles, reward_router_address)


class ProcessAgentLoopManager(AgentLoopManager):
    """
    AgentLoopManager that uses ProcessAgentLoopWorker instead of
    AgentLoopWorker, enabling per-step rm_scores for process GRPO.
    """

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ProcessAgentLoopWorker
        super().__init__(*args, **kwargs)