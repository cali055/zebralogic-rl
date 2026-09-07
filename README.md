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

## Project Structure
zebralogic_rl/
├── agent/                 # Multi-turn puzzle-solving agent and tool configuration
├── evaluation/            # Evaluation scripts and ZebraLogic evaluation data
├── merge/                 # Scripts for merging trained LoRA checkpoints
├── rl/
│   ├── outcome/           # Outcome-reward GRPO training
│   └── process/           # Process-reward GRPO training
└── sft/                   # Supervised fine-tuning

## Installation

Create and activate the environment, then install the required packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes the pinned dependencies used for the experiments, including the modified `verl` version used by this project.


## Data Setup

The repository includes the datasets required for training and evaluation.

### Supervised Fine-Tuning

- `sft/SFT_Train_final_split.parquet` — training data for SFT.
- `sft/SFT_Val_final.parquet` — validation data for SFT.

### Outcome-Reward GRPO

- `rl/outcome/FINALGRPOBaseline.parquet` — data used for the outcome-reward GRPO setup.

### Evaluation

- `evaluation/eval_zebralogic.parquet` — ZebraLogic evaluation dataset.

The provided training and evaluation scripts reference these datasets through their configured paths. Update the paths in the scripts or SLURM files.

### Outcome-Reward GRPO

The outcome-reward experiment uses single-turn GRPO, where the model generates a complete puzzle solution and receives an outcome-based reward based on the final solution.

Run the training job with:

```bash
sbatch rl/outcome/grpo_a100.sbatch
```

After training, merge the GRPO checkpoint:

```bash
sbatch merge/merge_grpo.sbatch
```

The resulting merged model can then be evaluated using the single-turn evaluation script:

```bash
python evaluation/eval_singleturn.py --model <model-path>
```

### Process-Reward GRPO

The process-reward experiments use a multi-turn setup in which the model solves puzzles incrementally through structured tool calls. The model is first initialized using supervised fine-tuning before process-reward GRPO training.

Run the SFT training job:

```bash
sbatch sft/run_multisft.sbatch
```

Merge the SFT checkpoint:

```bash
sbatch merge/merge_sft.sbatch
```

Run process-reward GRPO using either advantage formulation:

```bash
sbatch rl/process/grpo_process_advantagevanilla.sbatch
```

or:

```bash
sbatch rl/process/grpo_process_advantagecellnormalized.sbatch
```

After GRPO training, merge the resulting checkpoint:

```bash
sbatch merge/merge_grpo.sbatch
```

Finally, evaluate the trained model:

```bash
sbatch evaluation/eval_multiturn.sbatch
```
## Results

The experiments compared four training configurations:

| Configuration                                 | Cell Accuracy | Puzzle Accuracy |
| --------------------------------------------- | ------------: | --------------: |
| Single-turn GRPO (Outcome Reward)             |        84.37% |          82.90% |
| Multi-turn GRPO (Scalar Broadcast)            |        75.51% |          71.40% |
| Multi-turn GRPO (Cell-Identity Normalization) |        73.09% |          71.00% |
| Base-Initialized Cell-Identity GRPO           |        60.62% |          66.00% |

Under the experimental conditions of this thesis, the single-turn outcome-reward configuration achieved the best performance. Process rewards did not provide a measurable benefit over the outcome-reward baseline.

## Conclusion

The experiments show that process rewards did not provide a measurable improvement over outcome rewards for ZebraLogic puzzle solving under the conditions studied.

The single-turn outcome-reward GRPO configuration achieved the highest performance, while the multi-turn process-reward configurations performed lower. Supervised fine-tuning was necessary for the multi-turn setup to operate reliably, but it did not overcome the limitations of the process-reward approach.

Overall, the results suggest that providing finer-grained process feedback alone is not sufficient to improve performance when the model has not learned the underlying behaviors required for effective multi-step tool-based reasoning. The model has a general bias to solve these types of puzzles in a way due to which we have to use another model or bigger training budget.
