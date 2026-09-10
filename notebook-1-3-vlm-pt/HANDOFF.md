# Lab 3 — VLM Post-Training: handoff

Everything below describes the state on **tez-01-sig**, where this package already runs
end to end. Written 2026-09-02.

---

## 1. What this is

Lab 3 of the NVIDIA Singapore AI Day DLI workshop. Attendees post-train **Cosmos 3 Nano**
with LoRA on a labelled traffic-video dataset and measure what training bought them.

The day runs: **Lab 1** (deploy and prompt Cosmos 3 Reasoner, discover prompting has a
ceiling) → **SDG notebook** (generate synthetic video to fill dataset gaps) → **Lab 3**
(post-train and measure). The through-line: *prompting plateaus; post-training breaks
through, but only for classes you actually gave data for.*

Audience is mostly **first-time fine-tuners and non-expert coders**. Outputs are deliberately
plain: tables not JSON dumps, plain-English verdicts, no unexplained jargon.

## 2. Two repos, one source of truth

The lab lives in two GitLab repos. **TME is the source of truth — always edit there.**

| | repo | lab path | role |
|---|---|---|---|
| **source of truth** | `gitlab-master.nvidia.com/metropolis-tme/singapore-ai-day-dli` | `notebook-1-3-vlm-pt/` | internal build repo: generator, tools, HANDOFF, part-2 docs |
| **derived** | `gitlab.com/nvidia/dli/content/x-fx-138-v1` | `task1/lab/task/lab-3-vlm-pt/` | Singapore AI Day deployment; attendee subset only |

TME is source of truth because it holds everything prod does *not*: the generator that
writes the notebook (§7), the tools, this document, and the part-2 course docs. Prod holds
a strict 5-file subset, so it cannot round-trip an edit — a change made there is lost the
next time the generator runs.

Propagate TME to prod with `export_to_prod.sh` at the workspace root. It copies the five
attendee files, md5-verifies each, prints prod's `git status`, and **commits nothing**.

```
lab_3.ipynb  requirements.txt  configs/*.toml  helpers/dli_helpers.py
```

If a colleague edits the notebook directly in prod, do not copy the file back. Diff it
cell-by-cell and fold the intent into `tools/build_notebook.py`, then regenerate and
re-export. That was done once already, for eight DLI-environment cells (DinD data-volume
base path, credential gating, Lab-1 NIM teardown, no-pull image check, `BUILD_PTM=False`
shard validation, dataset clip validation, `scripts/` staging for the merge container).

The generator writes JSON with `ensure_ascii=False` and a trailing newline to match what
Jupyter saves, so browser-saved and generated notebooks diff cleanly.

## 3. Where everything is

```
notebook-1-3-vlm-pt/            the lab package (TME repo; synced to the box by sync.sh)
  lab_3.ipynb                   the lab — GENERATED, see §7
  lab_3.executed.ipynb          last full run, with outputs (1.6 MB)
  configs/                      train_config.toml, eval_config.toml
  helpers/dli_helpers.py        host helpers + the in-container merge entrypoint
  tools/                        prepare_dataset.py, execute_notebook.sh,
                                build_notebook.py, run_lab.sh
  start_jupyter.sh
  .env                          real NGC/HF tokens — NOT in git, NOT in the zip
  .env.example                  template

/mnt/dli/                       DLI_BASE — all data and outputs
  dataset/     2.7 G            assembled clips + annotations
  models/       17 G            converted Qwen3-VL checkpoint
  cache/        72 G            HuggingFace downloads
  work/         33 G            runs, evals, merged checkpoint, logs
  tao-skill-bank/               pinned 7.2.0 checkout
```

Container: `nvcr.io/nvidia/tao/tao-toolkit:7.2.0-cosmos-rl` (40.8 GB, already pulled).
GPUs: 2x H100 NVL, both free. Disk: `/` 189 G free, `/mnt` 115 G free.

> **`/mnt` is an Azure *temporary* disk** (`DATALOSS_WARNING_README.txt`). Everything under
> `/mnt/dli` can vanish on deallocate. Fine for development; the shipped DLI image must not
> depend on it.

## 4. How to run it

```bash
cd ~/lab-3-vlm-pt
DLI_GPU=0,1 ./tools/run_lab.sh ./tools/execute_notebook.sh lab_3.executed.ipynb
```

Interactive:

```bash
./start_jupyter.sh                       # binds 127.0.0.1:8888, token "lab3"
# from a laptop: ssh -A -L 8888:127.0.0.1:8888 drodge@tez-01-sig...
# open http://127.0.0.1:8888/lab?token=lab3, kernel "Python 3 (lab3)"
```

The kernel is `~/labenv` (`jupyter`, `nbconvert`, `tomli_w`, `matplotlib`). `ffmpeg` is
installed on the host and is **required** for the 10-second clip previews — without it the
trim silently returns full clips and the notebook balloons from 1.6 MB to ~147 MB.

## 5. Verified result

Full run, 2x H100, **9 minutes**, 30/30 cells, zero errors.

| stage | minutes |
|---|---|
| baseline evaluation | 1.1 |
| LoRA post-training | 4.3 |
| post-trained evaluation | 1.7 |
| adapter merge | 0.4 |
| merged evaluation | 1.3 |

| model | macro-F1 | precision | recall |
|---|---|---|---|
| base | 0.733 | 1.000 | 0.500 |
| base + LoRA adapter | 0.961 | 0.928 | 1.000 |
| merged checkpoint | 0.961 | 0.928 | 1.000 |

`collision` 32/32 → 32/32, `stalled` **0/32 → 32/32**. The merged checkpoint scores
identically to the adapter — that is the artifact VSS consumes, since VSS loads a plain
checkpoint rather than applying an adapter.

Per condition after post-training: `original` 1.000, `fog` 0.969, `night`/`rain` 0.938. The
five remaining errors are all false positives on negative clips in degraded conditions.

## 6. Notebook structure

```
1. Why Post-Training?        4. Dataset                    8. Compare Before vs After
2. Set Up the Environment      4.1 What Is In It             (by condition, then clip-by-clip)
   2.1 Deps and Base Dir       4.2 Look at the Data        9. Merge the Adapter for Deployment
   2.2 Credentials             4.3 Annotation Format      10. Tear Down
   2.3 GPU                     4.4 What Augmentation Adds     Review
   2.4 Docker / TAO         5. Baseline Evaluation
   2.5 Base Checkpoint         (+ where the base model fails)
   2.6 Dataset              6. Post-Training
3. Configuration            7. Post-Trained Evaluation
```

Design points worth preserving:

- **Terminology is "post-training"**, including identifiers (`RUN_POST_EVAL`, `post_metrics`,
  `dli_eval_posttrained`). `SFT` is kept only where it is the accepted name of the technique.
- **Configuration is flat constants** in section 3 (`LORA_R = 16`, `EPOCHS = 3`, …) applied
  onto `configs/*.toml`. An unknown key raises immediately rather than being ignored by the
  container an hour later.
- **Paths are relocatable.** A notebook has no `__file__`, so the first cell finds the lab
  folder by searching upward for `helpers/dli_helpers.py`. Verified from a moved copy and from
  a subdirectory. Do not reintroduce `Path.cwd()`-relative paths.
- **Credentials**: environment first, `.env` in the lab folder as fallback, values never
  printed. Empty entries (an unfilled `.env.example`) count as *not set*.
- **Two switches** choose what each stage reads: `TRAIN_ANNOTATIONS` / `EVAL_ANNOTATIONS`,
  either `AUGMENTED` (default) or `ORIGINAL`.

## 7. The notebook is generated

`tools/build_notebook.py` writes `lab_3.ipynb` deterministically. **Edit the builder and
regenerate — do not hand-edit the JSON**, it will be overwritten.

```bash
python3 tools/build_notebook.py        # -> lab_3.ipynb
```

Gotcha: the builder wraps cell text in `r"""..."""`. A `"""docstring"""` inside a code cell
terminates that block and breaks the build. Use `#` comments inside cells.

## 8. TAO 7.2.0 specifics that cost real debugging time

| symptom | cause and fix |
|---|---|
| `GET was unable to find an engine to execute this computation` | Qwen3-VL's non-overlapping Conv3D patch embedding has no cuDNN engine for this BF16 shape on driver 580. Set `policy.qwen3_vl_patch_embed` / `model.qwen3_vl_patch_embed` to `"linear"`. **`"auto"` does not select it.** This is also what lets the stock entrypoints be used, which keeps video decoding on the GPU |
| `ERROR: ... built for NVIDIA Driver Release 595.45 or later` | a version notice, not a failure. The run completes. The notebook says so |
| `No module named 'cosmos_rl.model_preparation'` | conversion must go through the TAO entrypoint `cosmos_rl.model_preparation.vlm_safetensors`, which dispatches to an isolated converter venv inside the image. Importing `cosmos_framework` directly fails by design |
| `AssertionError: Invalid role: None` | the Skill Bank training helper is a Cosmos-RL *dataset hook*, not a program — pass it to `cosmos-rl`, do not execute it |
| `'str' object has no attribute 'get'` (evaluate) | 7.2.0 evaluate schema: `task` and `metrics` must be tables, `vision.num_frames` (not `nframes`), `generation.max_tokens` (not `max_new_tokens`), `evaluation.answer_type` and `dataset.system_prompt` required, `metrics.names = []` for a classification task |
| `PermissionError: '/merged'` | evaluating an adapter merges it first; set `model.export_dir` or it writes beside the read-only adapter mount |
| `PermissionError: '/results/.inductor'` | every TAO container gets `HOME=/results`; a container without a `/results` mount needs a writable scratch dir |
| `KeyError: getpwuid()` / `No username set in the environment` | container runs as a bare uid — set `USER`/`LOGNAME` and pin the cache dirs |
| `training failed: 0` | zero-length dataset. The image ships two dataset hooks; the invoked one wants a JSON **list** of `{video, conversations}` |

**Known upstream defect.** The Skill Bank's `evaluate_video_conversation_single_gpu.py` is
broken against its own 7.2.0 image: it prepares vLLM-style inputs while
`cosmos_rl.evaluation.base.load_model` always constructs a `CosmosFrameworkRuntime` expecting
structured task dicts. We do not use it. Worth reporting to the TAO team.

## 9. Performance notes

Post-training went **12.6 → 8.3 → 4.3 min**:

| configuration | time |
|---|---|
| 1 GPU, Skill Bank helper (CPU decode) | 12.6 |
| 2 GPUs, same helper | 8.3 |
| 2 GPUs, no gradient checkpointing | 8.2 (no gain) |
| 2 GPUs, `mini_batch=4` | 8.1 (no gain) |
| **2 GPUs, `patch_embed=linear` + stock hook** | **4.3** |

The gain came from dropping the Skill Bank helper, which decoded video on **CPU under a global
lock**; the stock path uses the GPU decoder. The two memory levers changed nothing — the GPU
was starved, not saturated (33 GB of 95 GB used, utilisation dropping to 0%). If you tune
further, measure where the time goes before adding batch size.

## 10. Dataset

Three archives on Google Drive, **all private** (`gdown` gets HTTP 401), already assembled at
`/mnt/dli/dataset`:

| set | Drive ID |
|---|---|
| original clips (tar.gz) | `1RlKFBVypfF0Saj6gBviRujtj299jJUml` |
| augmented train | `1pvKVWRpSX4T3YnOlYW03DOQ_A7yEv5Ur` |
| augmented eval | `1XF8XTviDzkZQBOtbJkH8YbNHcWRekGVb` |

Layout: `train/` and `val/`, each with `videos/`, `annotations_original.json` and
`annotations_augmented.json` (a **superset** — originals plus fog/rain/night variants), plus
`dataset_info.json` carrying the task definition and prompts.

`tools/prepare_dataset.py` built it and is **not run by the lab**. Keep it: it is the only
record of how the dataset was assembled, and it normalises two things that differ between the
source archives — the augmented *train* clips sit in class folders with the condition in the
filename, the augmented *eval* clips sit in condition folders, and they use different
vocabularies (`nighttime`/`heavyrain` vs `night`/`rain`).

The 3-minute concatenated `*_eval.mp4` videos were **deliberately removed** — they are VSS
material, not training data.

Dataset decisions worth not undoing:

- **One label per whole clip; no window cutting.** The dataset labels a *clip*, not a moment.
  A window cut from the first seconds of a stalled-vehicle clip would be labelled `stalled`
  while showing a moving car. If per-event timestamps ever become available, windowing
  becomes possible — not before.
- **`train/` and `val/` follow the dataset authors' own split** (`train_clips` / `eval_clips`).
  Nothing is re-shuffled.
- **Augmented val clips are renamed `<clip>_<condition>.mp4`.** They arrive in condition
  folders with colliding basenames, and a flat `videos/` needs unique names.
- **Prompts are baked into the annotations**, with the task definition in `dataset_info.json`,
  so the notebook and the training data cannot drift apart. Change `TASK` in
  `tools/prepare_dataset.py` and re-run it, not the notebook.
- **Videos are hardlinked, not copied**, so assembling costs no extra disk on one filesystem
  (falls back to copying across devices).

## 11. Local validation — 10 Sep 2026, 1 × RTX PRO 6000 (96 GB), driver 590

Both deliverables were run end to end on the local box with
`DLI_BASE=~/singapore_aiday/data/lab3 DLI_GPU=1`, i.e. different paths and a single
GPU. Container times below; the Codex path adds 1–8 min of agent planning per step.

**Notebook** (nbconvert, fresh `lab3` kernel, 68 cells, zero error outputs):

| stage | run 1 | run 2 |
|---|---|---|
| baseline eval | 1.0 min | 0.8 min |
| LoRA training | 7.1 min | 6.6 min |
| post-trained eval | 1.8 min | 1.5 min |
| merge | 0.7 min | 0.5 min |
| merged eval | 1.0 min | 1.0 min |
| **whole notebook, nbconvert wall** | — | **10 min** |

Both runs: macro-F1 0.733 → 0.984 (+0.251), recall 1.000, precision 0.970 (2 FP), merged ≡ adapter.

**Part 2.3 walkthrough** driven by Codex (`gpt-5.5`, ChatGPT auth) with the TAO Skill
Bank plugin installed from the staged copy; prompts as on the docs page, paths adapted:

| step | container time | agent turn | outcome |
|---|---|---|---|
| 4 zero-shot eval | 41 s | — | first try; spec built from the skill; 0.733 / 1.000 / 0.500 |
| 5 plan (agent-chosen config) | — | — | plan produced; its recipe (batch 1, lr 2e-4, all 7 projections) **diverged** in training (val loss 4.6) |
| 6 train (explicit recipe) | 6 m 38 s | 674 s | val loss 0.024 / 0.017 / 0.020, best epoch_2; one self-corrected retry (`ModuleNotFoundError: cosmos_rl` — hook run as a script) |
| 7 eval + compare | 1 m 10 s | 463 s | recall 1.000, precision 0.955, 3 weather FPs, acc 0.977 |
| 8 merge + merged eval | 41 s + 57 s | 250 s | 17 GB / 4 shards to `models/Cosmos3-Nano-VLM-lora`; identical on all 128 responses |

Findings folded into the docs page: the Conv3D patch-embed failure without
`qwen3_vl_patch_embed=linear` (agent needed 3 attempts to find it), the disk-full
`basic_ios::clear: iostream error` (each FSDP checkpoint is 66 GB, two are kept —
the skill's 384 GiB gate is not decorative), the macro-F1 definition ambiguity
(3-class vs anomaly-vs-none), and the `cosmos_rl` hook row. The staged Skill Bank on
this box was initially an **empty directory tree**; the notebook's check now requires
`versions.yaml` and a real `SKILL.md`.

## 12. Open items

1. **Make the Drive links public**, or keep pre-staging the dataset. `§2.6` only verifies; it
   no longer downloads.
2. **Decide whether DLI pre-stages the converted checkpoint.** If yes, `§2.5` never runs for
   attendees and the `nvstaging` conversion image is irrelevant. If no, that image must be
   reachable from the student instance.
3. **`TRAIN_ANNOTATIONS = ORIGINAL` has not been run** — the clean-vs-augmented contrast is
   the one experiment still outstanding.
4. **`lab-3-vlm-pt-dli.zip`** (34 KB) is the barebones attendee subset: `lab_3.ipynb`,
   `configs/`, `helpers/`, `requirements.txt`, `.env.example`. Rebuild it if the lab changes.
5. `/mnt` is ephemeral — see §3.

## 13. Working agreement

- **Before every test run, delete the previous run's full FSDP checkpoints and
  intermediate outputs** (`**/checkpoints/*/checkpoints`, old `work/`, old `part2/runs`).
  Each FSDP checkpoint is 66 GB and two are kept; a full disk killed a run on 10 Sep.
  Keep only executed notebooks, `results.json` files, adapters and the merged model.

The laptop copy at `singapore-ai-day-dli/lab-3-vlm-pt/` was the source of truth, pushed with
`sync.sh`, which md5-verifies every file after rsync. **If you now edit on the box, the box
becomes the source of truth** — say so explicitly, because a stale local copy silently
overwrote a fix once during this work and cost a full run.
