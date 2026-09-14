# Lab 3 — VLM Post-Training

Post-train **Cosmos 3 Nano** with **LoRA** on a labelled video dataset, and measure the
difference on clips the model has never seen.

**Runtime:** ~9 minutes end to end on 2x H100 once the checkpoint and dataset are staged.
**Hardware:** one 80 GB GPU is enough; two shortens post-training.

## Layout

```
lab_3.ipynb              the lab — run top to bottom
configs/
  train_config.toml      full Cosmos-RL training spec
  eval_config.toml       full evaluator spec
helpers/
  dli_helpers.py         host-side helpers, and the in-container merge entrypoint
tools/
  prepare_dataset.py     one-time dataset assembly (not run by the lab)
  execute_notebook.sh    headless execution, writes lab_3.executed.ipynb
  build_notebook.py      regenerates lab_3.ipynb
start_jupyter.sh         launch JupyterLab with the lab environment
```

> `lab_3.ipynb` is **generated** by `tools/build_notebook.py`. Edit the builder and regenerate
> rather than hand-editing the JSON.

## Running it

```bash
export DLI_BASE=/mnt/dli        # everything reads and writes under here
export DLI_GPU=0,1              # "0", "1" or "0,1"
./start_jupyter.sh              # then open lab_3.ipynb, kernel "Python 3 (lab3)"
```

Headless, keeping outputs:

```bash
DLI_BASE=/mnt/dli DLI_GPU=0,1 tools/execute_notebook.sh
```

## Configuration

Section 3 of the notebook exposes the settings worth changing as flat constants:

```python
LORA_R          = 16
LORA_ALPHA      = 32
LORA_TARGETS    = ["q_proj", "v_proj"]
EPOCHS          = 3
BATCH           = 4
LR              = 1e-4
```

These are applied on top of the TOML files in `configs/`. An unknown key raises immediately
rather than being ignored by the container later.

Two switches choose what each stage reads:

```python
TRAIN_ANNOTATIONS = AUGMENTED   # or ORIGINAL
EVAL_ANNOTATIONS  = AUGMENTED   # or ORIGINAL
```

## Environment

| variable | default | meaning |
|---|---|---|
| `DLI_BASE` | notebook's parent | root for dataset, models, cache, work |
| `DLI_GPU` | first GPU | `0`, `1` or `0,1` |
| `DLI_DATA` | `$DLI_BASE/dataset` | assembled dataset |
| `DLI_PTM` | `$DLI_BASE/models/Cosmos3-Nano-VLM` | converted base checkpoint |
| `DLI_ROOT` | `$DLI_BASE/work` | runs, evaluations, logs |
| `NGC_KEY` / `HF_TOKEN` | — | container pull, model download |

## Expected result

| model | macro-F1 | precision | recall |
|---|---|---|---|
| base | 0.733 | 1.000 | 0.500 |
| base + LoRA adapter | 0.961 | 0.928 | 1.000 |
| merged checkpoint | 0.961 | 0.928 | 1.000 |

`collision` 32/32 -> 32/32, `stalled` **0/32 -> 32/32**. The merged checkpoint is the artifact
a deployment target such as VSS consumes, since it loads a plain model rather than an adapter.

## Prerequisites

- TAO container `nvcr.io/nvidia/tao/tao-toolkit:7.2.0-cosmos-rl` (~41 GB on disk)
- Cosmos 3 Nano converted to Qwen3-VL safetensors (~17 GB) — section 2.5 builds it if absent
- `ffmpeg` on the host, for the 10-second clip previews
- Driver >= 580, CUDA >= 13.0
- >= 150 GB free where `DLI_ROOT` points: training keeps two FSDP checkpoints (~66 GB each),
  reclaimed by the tear-down cell

## Notes that cost real debugging time

| symptom | cause |
|---|---|
| `GET was unable to find an engine to execute this computation` | Qwen3-VL's Conv3D patch embedding has no cuDNN engine for this BF16 shape. Set `qwen3_vl_patch_embed = "linear"`; `"auto"` does not select it |
| `ERROR: ... built for NVIDIA Driver Release 595.45 or later` | a version notice, not a failure. The run completes normally |
| `KeyError: getpwuid()` / `No username set in the environment` | the container runs as a bare uid; `USER`/`LOGNAME` and pinned cache dirs fix it |
| `PermissionError: './results'` | `_get_results_dir()` returns a relative path unless `TAO_API_JOB_ID` is set |
| `training failed: 0` | zero-length dataset. The image ships two dataset hooks; the invoked one wants a JSON **list** of `{video, conversations}` |
| `'str' object has no attribute 'get'` | evaluator config shape: `task` and `metrics` must be tables, and `dataset.system_prompt` must be set |
| `PermissionError: '/merged'` | evaluating an adapter merges it first; without `model.export_dir` it writes beside the read-only adapter mount |
