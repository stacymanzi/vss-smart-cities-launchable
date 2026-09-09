# Lab 3 — VLM Post-Training

Fine-tune **Cosmos 3 Nano** with **LoRA** to detect traffic anomalies, and measure
the difference on clips the model has never seen.

**Runtime:** ~20 minutes end to end (verified). **Hardware:** 1× 80 GB GPU.

| File | What it is |
|---|---|
| `lab_3.ipynb` | the lab — run top to bottom |
| `dli_helpers.py` | plumbing (video cutting, docker invocation, scoring) |
| `build_notebook.py` | regenerates the notebook deterministically |
| `PROMPTS.md` | the winning prompts, fully resolved and copy-pasteable |

## What attendees do

1. **Setup** — verify GPU, Docker, credentials, base checkpoint
2. **Dataset** — cut labelled windows from traffic footage; split without leakage
3. **Zero-shot eval** — measure the untuned model (the number to beat)
4. **LoRA fine-tune** — train adapters with NVIDIA TAO (Cosmos-RL), ~3 min
5. **Final eval** — same held-out clips, same metric
6. **Compare** — per-class results and what the gap means

## Result they should reproduce

| stage | macro-F1 | P | R |
|---|---|---|---|
| zero-shot | 0.666 | 1.000 | 0.423 |
| **LoRA fine-tuned** | **0.894** | 1.000 | 0.808 |

`stalled_vehicle` goes **0/11 → 10/11** on a stalled clip the model never trained on —
genuine transfer, not memorisation. `wrong_way` stays 0/4 because it has only one
training clip: the lab's closing point is that **fine-tuning transfers where you gave
it data, and nowhere else.**

## Prerequisites (pre-staged on the DLI instance)

```bash
export NGC_KEY=...     # nvcr.io images
export HF_TOKEN=...    # gated Cosmos3 repos
```

- TAO container `nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-rl` (~28 GB)
- Cosmos 3 Nano converted to Qwen3-VL safetensors (~17 GB)
- ≥150 GB free disk — training keeps 2 FSDP checkpoints (~66 GB each); the
  cleanup cell reclaims them
- Traffic clips in `data/`

Paths are overridable via `DLI_GPU`, `DLI_PTM`, `DLI_DATA`, `DLI_ROOT`.

## Gotchas already handled in the notebook

Each of these was hit during development and produces a misleading error:

| Symptom | Real cause |
|---|---|
| `KeyError: getpwuid(): uid not found` | host uid has no passwd entry in the container; torch's `cache_dir()` calls `getpass.getuser()` |
| `PermissionError: './results'` | `_get_results_dir()` returns a **relative** path unless `TAO_API_JOB_ID` is set |
| `FileNotFoundError: model-0000x.safetensors` | **not missing** — weights are mode 600 root-owned while configs are 644, so config load succeeds and only weights fail |
| `training failed: 0` | zero-length dataset → division by zero. The image ships **two** dataset hooks with different schemas; the invoked one wants classic **llava** (a JSON *list* of `{video, conversations}`) |
| `AssertionError: Compile is not supported for HFModel` | `train.compile` defaults to True when the key is omitted — it must be explicitly `False` |
| `FileNotFoundError: .rank_0_complete` | `ckpt.max_keep=1` races the async completion-marker writer; use 2 and delete checkpoints afterwards |
| `'str'/'list' object has no attribute 'get'` | in the evaluate TOML, `task` must be `{"type": ...}` and `metrics` `{"names": [...]}` — dicts, not bare values |

## Regenerating the notebook

```bash
python3 build_notebook.py     # rewrites lab_3.ipynb
```

Edit `build_notebook.py`, not the `.ipynb`, so the lab stays reproducible.
