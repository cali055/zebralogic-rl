# Training Large Language Models to Solve Logic Puzzles using Reinforcement Learning

Master's thesis project at TU Braunschweig (IFN). Investigates whether process rewards provide a measurable benefit over outcome rewards when training large language models to solve logic grid puzzles.

## Overview

Large language models have demonstrated strong performance on many natural language tasks, but structured reasoning problems remain challenging when solving a task requires maintaining consistency across many interdependent deductions.

This thesis investigates whether reinforcement learning with verifiable rewards can improve the ability of large language models to solve logic grid puzzles. In particular, it studies whether **process rewards**, which provide feedback after individual deduction steps,
offer an advantage over **outcome rewards**, which provide a single reward for the completed solution.

The experiments use ZebraLogic puzzles and compare four training configurations:

1. **Single-turn GRPO** — the model generates a complete solution in one generation pass and receives an outcome-based reward.

2. **Multi-turn GRPO with scalar-broadcast advantages** — the model solves the puzzle incrementally through structured tool calls, with process rewards assigned to individual cell insertions.

3. **Multi-turn GRPO with cell-identity normalization** — the same multi-turn setup using a different advantage computation scheme, where rewards are normalized according to the identity of the cell being inserted.

4. **Base-initialized cell-identity GRPO** — the cell-identity formulation trained directly from the base model without supervised fine-tuning initialization.

The multi-turn configurations use an external grid tool that allows the
model to modify the puzzle state one cell at a time. This provides a
discrete and verifiable state transition for each reasoning step.

The experiments investigate two main questions:

- Does reinforcement learning improve logic puzzle solving as puzzle
  complexity increases?
- Do process rewards provide a measurable benefit over outcome rewards?

The results show that, under the experimental conditions of this
thesis, the single-turn outcome-based GRPO configuration outperformed
all process-based configurations. Process rewards therefore did not
provide a measurable performance benefit, although supervised
initialization substantially improved the ability of the model to
operate in the multi-turn environment.
