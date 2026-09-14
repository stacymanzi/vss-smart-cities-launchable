#!/usr/bin/env python3
"""Generate the Cosmos 3 Nano LoRA post-training notebook.

Kept as a builder script (rather than hand-edited JSON) so the notebook can be
regenerated deterministically after any change to the flow.

The lab is written to be **use-case agnostic**: the task, its label space and its
prompts live in a single TASK cell, and the dataset loader reads any
folder-per-class video layout. The traffic-anomaly dataset is the worked example
and is named only where the dataset itself is discussed.
"""
import json
from pathlib import Path

# this script lives in tools/; the notebook belongs one level up in the lab dir
OUT = Path(__file__).resolve().parent.parent / "lab_3.ipynb"
cells = []


def _cid():
    # nbformat >=4.5 requires a cell id; without it jupyter emits a
    # MissingIDFieldWarning that will become a hard error.
    return f"c{len(cells):03d}"


def md(text):
    cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "id": _cid(), "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ===== 0. Intro =====
md(r"""
# Lab 3: Post-Training Vision Language Models

## Cosmos 3 Nano + LoRA

In **Lab 1** you deployed Cosmos 3 Reasoner and learned to prompt it — and saw that prompting
alone has a ceiling. In the **SDG notebook** you generated synthetic footage to fill the gaps
in a dataset. This notebook closes the loop: you will **post-train** Cosmos 3 Nano with
**LoRA** on your own labelled video, and measure exactly what training bought you on clips
the model has never seen.

A video classification task is defined by three things: a set of labels, a prompt, and one
folder of clips per label. Substituting those retargets the notebook to a different problem.
The worked example is a traffic-anomaly dataset; no other section depends on that choice.

The model answers one line per clip:

```
LABEL=<one of your classes, or none>
```

## What to Expect

We follow a practical improvement loop:

```mermaid
flowchart TD
    base["Baseline Evaluation"] --> sft["LoRA Supervised Post-Training"]
    sft --> ftEval["Post-Trained Evaluation"]
    ftEval --> gap["Gap Analysis by class and condition"]
    gap --> next["Specification for the next data run"]
```

1. Inspect the dataset and review representative clips.
2. Evaluate the **base** model on held-out clips to establish a baseline.
3. Fine-tune with LoRA on the training split.
4. Evaluate the tuned model on the **same** clips and compare.
5. Analyse the result by class and by capture condition to identify remaining gaps.

Each stage reads one of two annotation files: `annotations_augmented.json` (originals plus
fog, rain and night variants; the default) or `annotations_original.json` (originals only).
Section 3 provides one switch for training and one for evaluation.

### Learning Objectives:
This notebook explores the following topics:
* Why post-training is needed once prompt engineering runs out of room
* How to read a prepared video dataset, and why you watch the clips before training
* How the choice of **task** — classification, binary, multiple-choice or caption — fixes
  your annotations, your metric, and what the model can learn
* Why **macro-F1** is the right metric for an imbalanced detection task, and why a
  **precision drop** can accompany a better model
* Establishing a zero-shot baseline *before* training anything
* **LoRA** post-training with Cosmos-RL: the spec keys that matter, how to
  choose their values, and what breaks when you move them
* Reading validation loss vs. real task accuracy — and why they disagree
* Evaluating the tuned checkpoint on held-out clips and comparing before vs after
* Gap analysis by class **and by condition**, and what augmented data does and does not buy

> **Runtime:** roughly 25 minutes end to end once setup is complete, on two H100 GPUs.
> Measured stage times are reported in section 9.
""")

md(r"""
### Table of Contents

**[1. Why Post-Training?](#1.-Why-Post-Training?)**  
**[2. Set Up the Environment](#2.-Set-Up-the-Environment)**  
**[3. Configuration](#3.-Configuration)**  
**[4. Dataset](#4.-Dataset)**  
&nbsp;&nbsp;&nbsp;&nbsp;[4.1 What Is In It](#4.1-What-Is-In-It)  
&nbsp;&nbsp;&nbsp;&nbsp;[4.2 Look at the Data](#4.2-Look-at-the-Data)  
&nbsp;&nbsp;&nbsp;&nbsp;[4.3 Annotation Format](#4.3-Annotation-Format)  
&nbsp;&nbsp;&nbsp;&nbsp;[4.4 What Augmentation Adds](#4.4-What-Augmentation-Adds)  
&nbsp;&nbsp;&nbsp;&nbsp;[4.5 How the Task Is Posed](#4.5-How-the-Task-Is-Posed-—-and-How-Else-It-Could-Be)  
**[5. Baseline Evaluation](#5.-Baseline-Evaluation)**  
**[6. Post-Training](#6.-Post-Training)**  
&nbsp;&nbsp;&nbsp;&nbsp;[6.1 Training Configuration](#6.1-Training-Configuration)  
&nbsp;&nbsp;&nbsp;&nbsp;[6.2 Train](#6.2-Train)  
&nbsp;&nbsp;&nbsp;&nbsp;[6.3 Validation Loss](#6.3-Validation-Loss)  
**[7. Post-Trained Evaluation](#7.-Post-Trained-Evaluation)**  
**[8. Compare Before vs After](#8.-Compare-Before-vs-After)**  
**[9. Merge the Adapter for Deployment](#9.-Merge-the-Adapter-for-Deployment)**  
**[10. Tear Down](#10.-Tear-Down)**  
**[Review](#Review)**
""")

# ══════════════════════════════════════════════════════════ 1. Why
md(r"""
---
### 1. Why Post-Training?

Lab 1 showed Cosmos 3 answering questions about video without any training. Post-training is
warranted when prompt engineering reaches a measurable ceiling.

On a narrow detection task, a base VLM is characteristically **high-precision and
low-recall**: predictions it does make are usually correct, but most positives are missed.
Prompt revisions — enumerating the classes, describing the scene, requesting a confidence
score, adding a reasoning step — typically do not change this. The model is not
misinterpreting the question; it is under-predicting the positive class.

Prompting changes what the model is asked. Post-training changes what the model has seen.

#### Full post-training vs LoRA

| | Full post-training (dense SFT) | **LoRA** |
|---|---|---|
| what updates | every weight | small low-rank adapters |
| optimizer state | for all 8B params | for ~0.1% of them |
| artifact | a new 17 GB model | a **15 MB** adapter |
| risk | catastrophic forgetting | base weights frozen |

LoRA inserts a pair of low-rank matrices alongside selected attention projections and trains
only those. Because the base weights stay frozen, no gradients or optimizer moments are
allocated for them, which also reduces step time.

#### The loop this notebook runs

```
Baseline evaluation  ->  LoRA post-training (SFT)  ->  Post-trained evaluation  ->  Gap analysis
```

Both evaluations use the same held-out clips and the same metric, so the difference is
attributable to training alone. The closing gap analysis indicates which data to collect next.
""")

# ══════════════════════════════════════════════════════════ 2. Setup
md(r"""
### 2. Set Up the Environment

Everything in this section is a one-time cost. On a DLI instance most of it is pre-staged and
the cells simply verify; on a fresh machine each cell does the work.

Run the section top to bottom and stop at the first `FAIL`.

#### 2.1 Python Dependencies and Base Directory

The notebook itself needs very little — the heavy lifting happens inside the Cosmos-RL container.

All inputs and outputs live under a single **base directory**, set with `DLI_BASE`:

```
$DLI_BASE/
  dataset/          the prepared clips and annotations
  models/           the converted Qwen3-VL checkpoint (~17 GB)
  cache/            HuggingFace downloads (~50 GB)
  tao-skill-bank/   pinned Skill Bank checkout
  work/             runs, evaluations, merged checkpoint, logs
```

It defaults to the lab folder's parent. Individual `DLI_*` variables still override any single
location when it has to sit on a different mount.

Every path in the notebook is derived from the lab folder, which is located by searching for
`helpers/dli_helpers.py` at or above the working directory — so the folder can be moved or
renamed and the notebook still runs.
""")

code(r"""
import os, sys, subprocess, json, time, shutil
from pathlib import Path

# Locate the lab directory by looking for a file we know ships with it, walking
# up from the working directory. A notebook has no __file__, and the kernel's cwd
# depends on how it was started. Anchoring on a marker keeps every path relative,
# so the whole folder can be moved or renamed and still run.
def _find_lab_dir(marker="helpers/dli_helpers.py"):
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / marker).exists():
            return candidate
    raise RuntimeError(
        f"could not locate the lab directory: no {marker} at or above {here}")

LAB_DIR    = _find_lab_dir()
HELPER_DIR = LAB_DIR / "helpers"
CONFIG_DIR = LAB_DIR / "configs"
TOOLS_DIR  = LAB_DIR / "tools"
sys.path.insert(0, str(HELPER_DIR))
import dli_helpers as H

print(f"lab directory: {LAB_DIR}")

# Everything this lab reads or writes lives under one directory. In the lab that is
# the course data volume: the notebook container and the Docker daemon that runs
# the Cosmos-RL container both mount it at /dli/task/data, and a `docker run -v` source
# is resolved by the daemon, so this is the only place both sides can see. The
# base checkpoint and the dataset are staged there before the lab starts; runs
# are written next to them. DLI_BASE overrides this on other machines, and the
# individual DLI_* variables still override a single location.
_default_base = Path("/dli/task/data/lab3") if Path("/dli/task/data").is_dir() else LAB_DIR.parent
BASE = Path(os.environ.get("DLI_BASE", _default_base)).resolve()
BASE.mkdir(parents=True, exist_ok=True)
print(f"base directory: {BASE}")

# tomli_w writes the Cosmos-RL spec; matplotlib draws the comparison. Both are installed
# in the lab image. Everything heavy runs inside the Cosmos-RL container.
import tomli_w, matplotlib
print("dependencies ready")
print(f"  python     {sys.version.split()[0]}")
# ffmpeg is optional: the dataset ships pre-cut, so nothing here transcodes.
print(f"  ffmpeg     {'found' if shutil.which('ffmpeg') else 'not installed (optional)'}")
""")

md(r"""
#### 2.2 Credentials

Two tokens are used:

| variable | needed for |
|---|---|
| `NGC_KEY` | pulling the TAO container from `nvcr.io` |
| `HF_TOKEN` | downloading the base checkpoint from HuggingFace |

They are read from the environment first. Anything missing is filled in from a `.env` file in
the lab folder, so exporting them in your shell always takes precedence:

```bash
export NGC_KEY=...
export HF_TOKEN=...
```

The cell reports only *where* each value came from — never the value itself. On a DLI instance
where the container and checkpoint are already staged, neither is required.
""")

code(r"""
# In the lab the Cosmos-RL image and the base checkpoint are already staged, so no
# credential is needed. On another machine, NGC_KEY (to pull the image) and
# HF_TOKEN (to build the checkpoint) are read from the environment, then from a
# .env file next to the notebook. Values are never printed.
if os.environ.get("DLI_REQUIRE_CREDENTIALS") == "1":
    H.credential_status(["NGC_KEY", "HF_TOKEN"], env_file=LAB_DIR / ".env")
else:
    found = [n for n in ("NGC_KEY", "HF_TOKEN") if os.environ.get(n)]
    print("credentials: not required in this lab (image and checkpoint are pre-staged)"
          + (f"; present in the environment: {', '.join(found)}" if found else ""))

""")

md(r"""
#### 2.3 GPU

Set `GPUS` to the device(s) this notebook may use — a single index or a list. Training
shards across every device listed, and evaluation uses the same set.

Cosmos 3 Nano LoRA needs roughly **60 GB on one device**, so a single 80 GB card
(H100 / A100-80G / RTX PRO 6000) is sufficient. Listing two devices shortens training but is
not required.
""")

code(r"""
# Lab 1's model server runs as a container on the same Docker daemon and may still
# hold a GPU. Remove it first; this is a no-op when it is already gone.
subprocess.run(["docker", "rm", "-f", os.environ.get("LAB1_NIM_CONTAINER", "cosmos3-reasoner")],
               capture_output=True)

gpus = H.gpu_table()
for g in gpus:
    print(f"  GPU {g['index']}: {g['name']}  {g['used_gb']:.0f} / {g['total_gb']:.0f} GB used")

# Devices this notebook may use. Default: every GPU on the machine - the lab VM has
# two H100s and training shards across both. DLI_GPU overrides ("0" or "0,1"). The
# daemon that runs the Cosmos-RL container sees the same devices with the same indices.
GPUS = H.as_gpu_list(os.environ.get("DLI_GPU", ",".join(str(g["index"]) for g in gpus)))
print(f"\nUsing GPU(s): {GPUS}")

# Look each card up by its nvidia-smi index, not by position in the list — the
# two only coincide when indices start at 0 and are contiguous.
by_index = {g["index"]: g for g in gpus}
missing = [g for g in GPUS if g not in by_index]
assert not missing, f"GPU(s) {missing} not found; available: {sorted(by_index)}"

# Memory held by a container that was just removed takes a few seconds to disappear
# from nvidia-smi, so poll briefly before deciding a GPU is genuinely occupied.
def _free_gb(g):
    return by_index[g]["total_gb"] - by_index[g]["used_gb"]

for _ in range(6):
    if all(_free_gb(g) >= 60 for g in GPUS):
        break
    time.sleep(5)
    by_index = {g["index"]: g for g in H.gpu_table()}

enough = True
for g in GPUS:
    enough &= H.check(f"GPU {g} has >= 60 GB free", _free_gb(g) >= 60, f"{_free_gb(g):.0f} GB free")
if not enough:
    raise RuntimeError("A GPU is still occupied - most likely the Lab 2 kernel. In JupyterLab open "
                       "'Running Terminals and Kernels' and shut the other notebooks down, then re-run this cell.")
""")

md(r"""
#### 2.4 Docker and the Cosmos-RL Container

Checkpoint preparation, training and evaluation all run inside the **Cosmos-RL** container (image `tao-toolkit:7.2.0-cosmos-rl`)
container (~41 GB on disk), which is normally pre-staged on the DLI instance. The cell pulls
it if it is absent.

A single image covers all three stages: it packages the checkpoint-conversion entrypoint alongside an
isolated Cosmos-Framework converter environment, so no second image is required.

Qwen3-VL embeds video patches with a non-overlapping Conv3D. On some driver and cuDNN
combinations that BF16 convolution has no selectable engine and fails with
`RuntimeError: GET was unable to find an engine to execute this computation`. Because the
patches do not overlap, an unfold-plus-linear formulation is mathematically identical.

Cosmos-RL exposes this directly: `policy.qwen3_vl_patch_embed` for training and
`model.qwen3_vl_patch_embed` for evaluation, each accepting `auto`, `linear` or `conv3d`.
Both configurations below request `linear` explicitly, because `auto` does not select it on
this driver. With that set, the stock training hook and the stock evaluator are used, which
keeps video decoding on the GPU.

Two filesystems are involved and they are frequently distinct — the image is written to
Docker's storage directory, while checkpoints are written under `DLI_ROOT`. The cell checks
both.
""")

code(r"""
H.check("docker daemon reachable", H.docker_ok(), os.environ.get("DOCKER_HOST", "local socket"))

# Check the filesystem that holds the checkpoint, the dataset and every run:
# training keeps two FSDP checkpoints (~66 GB each).
WORK_FS = Path(os.environ.get("DLI_ROOT", BASE))
while not WORK_FS.exists():
    WORK_FS = WORK_FS.parent
free_disk = shutil.disk_usage(WORK_FS).free / 1e9
H.check(f"disk >= 150 GB free on {WORK_FS}", free_disk >= 150,
        f"{free_disk:.0f} GB - training keeps 2 FSDP checkpoints (~66 GB each). "
        f"Set DLI_ROOT to a bigger mount if this fails.")

# The Cosmos-RL image is seeded into the course's Docker daemon when the environment
# starts, so nothing is pulled here. Pulling needs an NGC key and ~40 GB of disk,
# which only makes sense on a development machine.
if not H.image_present() and os.environ.get("NGC_KEY"):
    print(f"\n{H.IMAGE} not found - NGC_KEY is set, pulling (development machines only, ~40 GB) ...")
    H.pull_image()
if not H.check(f"Cosmos-RL image present ({H.IMAGE})", H.image_present()):
    raise RuntimeError(f"{H.IMAGE} is not in the Docker daemon this notebook talks to. "
                       "In the lab it is pre-seeded at launch (COURSE_EXTRA_IMAGES in the course compose); ask a TA.")

""")

md(r"""
#### 2.5 Base Checkpoint

Cosmos-RL cannot load the native `cosmos3_omni` checkpoint format, so training uses a **Qwen3-VL
safetensors** conversion of Cosmos 3 Nano (~17 GB). On a DLI instance this is pre-staged and
the cell below verifies it; otherwise the cell produces it.

Conversion downloads `nvidia/Cosmos3-Nano` and merges its language-model tensors onto the
`Qwen/Qwen3-VL-8B-Instruct` architecture. The base Nano release also ships a Qwen3-VL-shaped
visual tower, so its trained vision encoder is carried over as well.

Conversion runs in the **same image as training**, through the packaged entrypoint
`cosmos_rl.model_preparation.vlm_safetensors`. That entrypoint dispatches to a
Cosmos-Framework converter pinned in an isolated environment inside the image; importing
`cosmos_framework` from the default interpreter therefore fails by design.

The cell also clones the **TAO Skill Bank**, the packaged reference for the Cosmos-Reason
workflows, which Part 2 of the day uses.

Expect roughly **20–40 minutes** and ~50 GB of HuggingFace cache on a cold run. Set
`BUILD_PTM = False` to verify only.
""")

code(r"""
# PTM = pre-trained model: the base checkpoint that fine-tuning starts from.
BUILD_PTM = False    # the converted checkpoint is staged with the course data. True rebuilds it
                     # from HuggingFace (needs HF_TOKEN, network, ~50 GB and 20-40 min); only
                     # needed when running outside the course environment.

PTM        = Path(os.environ.get("DLI_PTM",        BASE / "models" / "Cosmos3-Nano-VLM"))
CACHE_DIR  = Path(os.environ.get("DLI_CACHE",      BASE / "cache"))
SKILL_BANK = Path(os.environ.get("DLI_SKILL_BANK", BASE / "tao-skill-bank"))

# The TAO Skill Bank is used again in Part 2 of the day. If the staged copy is missing
# or incomplete, the pinned tag is cloned next to it. This notebook does not depend on it.
SKILL_BANK = H.ensure_skill_bank(SKILL_BANK)
H.check(f"TAO Skill Bank {H.SKILL_BANK_REF}", H.skill_bank_complete(SKILL_BANK), str(SKILL_BANK))

if BUILD_PTM and not sorted(PTM.glob("*.safetensors")):
    H.ensure_ptm(PTM, CACHE_DIR, SKILL_BANK)

# Every shard named by the index must be present and readable; raises with guidance if not.
H.verify_checkpoint(PTM)
""")

md(r"""
#### 2.6 Dataset

The lab expects an **already-assembled** dataset directory. Point `DLI_DATA` at it:

```
dataset/
  train/videos/*.mp4
  train/annotations_original.json      originals only
  train/annotations_augmented.json     originals + weather/lighting variants
  val/videos/*.mp4
  val/annotations_original.json
  val/annotations_augmented.json
  dataset_info.json                    task metadata + counts
```

On a DLI instance this is staged with the course data. The cell below only **verifies**
what is there; it downloads nothing.
""")

code(r"""
DATASET = Path(os.environ.get("DLI_DATA", BASE / "dataset")).resolve()

required = ["dataset_info.json",
            "train/annotations_original.json", "train/annotations_augmented.json",
            "val/annotations_original.json",   "val/annotations_augmented.json"]
missing = [f for f in required if not (DATASET / f).exists()]
if not missing:
    # The loader copies clip by clip: require every video the annotations point at.
    for split in ("train", "val"):
        recs = H.load_json(DATASET / split / "annotations_augmented.json")
        absent = [r["video"] for r in recs if not (DATASET / split / r["video"]).exists()]
        if absent:
            missing.append(f"{split}: {len(absent)} of {len(recs)} clips referenced by the annotations")

print(f"dataset root: {DATASET}\n")
if missing:
    H.check("dataset present", False, f"missing {missing}")
    raise RuntimeError(f"The dataset under {DATASET} is incomplete. The course loader stages it before the lab "
                       "starts - wait for it to finish or ask a TA. (Elsewhere: set DLI_DATA to an assembled dataset.)")
else:
    for split in ("train", "val"):
        n_vid = len(list((DATASET / split / "videos").glob("*.mp4")))
        n_org = len(H.load_json(DATASET / split / "annotations_original.json"))
        n_aug = len(H.load_json(DATASET / split / "annotations_augmented.json"))
        print(f"  {split:<6} {n_vid:>4} videos   {n_org:>4} original   {n_aug:>4} augmented")
    H.check("dataset present", True, str(DATASET))
""")

# ══════════════════════════════════════════════════════════ 3. Config
md(r"""
---
### 3. Configuration

Paths, sampling parameters and LoRA hyperparameters, together with the two switches that
select the annotation file for each stage.

The default configuration trains and evaluates on `annotations_augmented.json`, which contains
the originals and their weather variants. Setting either switch to
`annotations_original.json` runs the originals-only comparison.
""")

code(r"""
WORK = Path(os.environ.get("DLI_ROOT", BASE / "work")).resolve()
WORK.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATASET / "train"
VAL_DIR   = DATASET / "val"
RUNS  = WORK / "runs"       # training output (adapters)
EVALS = WORK / "evals"      # evaluation output
LOGS  = WORK / "logs"
for d in (RUNS, EVALS, LOGS): d.mkdir(parents=True, exist_ok=True)

# ---- which annotations each stage reads --------------------------------
AUGMENTED = "annotations_augmented.json"   # originals + fog / rain / night
ORIGINAL  = "annotations_original.json"    # originals only

TRAIN_ANNOTATIONS = AUGMENTED    # switch to ORIGINAL to train on clean clips only
EVAL_ANNOTATIONS  = AUGMENTED    # switch to ORIGINAL to score on clean clips only

info = H.load_json(DATASET / "dataset_info.json")

# Full configurations ship in configs/.
TRAIN_TOML = CONFIG_DIR / "train_config.toml"
EVAL_TOML  = CONFIG_DIR / "eval_config.toml"

print(f"work   {WORK}")
print(f"ptm    {PTM}")
print(f"GPU(s) {GPUS}")
print(f"train on {TRAIN_ANNOTATIONS}")
print(f"eval  on {EVAL_ANNOTATIONS}")
""")

md(r"""
#### The settings you are likely to change

The full configurations live in `configs/train_config.toml` and `configs/eval_config.toml`;
they carry dozens of keys that are already correct. These are the ones worth touching.
""")

code(r"""
# ---- model ---------------------------------------------------------------
BASE_MODEL      = "/ptm"                    # base checkpoint, as mounted in the container
MAX_LENGTH      = 40960                     # do not lower: video inputs need the room

# ---- data ----------------------------------------------------------------
NUM_FRAMES      = 8                         # frames sampled per clip, train and eval alike
ANSWER_TYPE     = "freeform"                # answers are a LABEL= line, not A/B/C/D

# ---- LoRA ----------------------------------------------------------------
LORA_R          = 16
LORA_ALPHA      = 32
LORA_TARGETS    = ["q_proj", "v_proj"]       # supported: q_proj k_proj v_proj o_proj gate_proj up_proj
                                            #   down_proj attn.qkv attn.proj, or "all-linear"
EPOCHS          = 3
BATCH           = 4                         # must be a multiple of mini_batch (=1)
LR              = 1e-4

# ---- what each stage reads -----------------------------------------------
TRAIN_ANNOTATIONS = AUGMENTED               # or ORIGINAL, to train on clean clips only
EVAL_ANNOTATIONS  = AUGMENTED               # or ORIGINAL, to score on clean clips only

SYSTEM_PROMPT   = info["task"]["system"]
""")

md(r"""
Those settings are applied on top of the shipped TOML files. The mapping below is mechanical —
it exists so a typo is caught here rather than by the container an hour later.
""")

code(r"""
TRAIN_SETTINGS = {
    "policy.model_name_or_path":            BASE_MODEL,
    "policy.model_max_length":              MAX_LENGTH,
    "policy.parallelism.dp_shard_size":     len(GPUS),
    "policy.lora.r":                        LORA_R,
    "policy.lora.lora_alpha":               LORA_ALPHA,
    "policy.lora.target_modules":           LORA_TARGETS,
    "train.epoch":                          EPOCHS,
    "train.train_batch_per_replica":        BATCH,
    "train.optm_lr":                        LR,
    "custom.vision.nframes":                NUM_FRAMES,
    "custom.system_prompt":                 SYSTEM_PROMPT,
    "custom.train_dataset.annotation_path": f"/train/{TRAIN_ANNOTATIONS}",
    "custom.val_dataset.annotation_path":   f"/val/{EVAL_ANNOTATIONS}",
}

EVAL_SETTINGS = {
    "model.model_name":        BASE_MODEL,
    "model.base_model_path":   BASE_MODEL,
    "model.enable_lora":       False,       # set per run when scoring an adapter
    "model.max_length":        MAX_LENGTH,
    # Tensor parallelism splits one model across GPUs; an 8B model in bf16 (~17 GB)
    # fits on a single GPU, so it stays at 1. Extra GPUs are used through num_gpus
    # (data parallel, one copy of the model per GPU) below.
    "model.tp_size":           1,
    "vision.num_frames":       NUM_FRAMES,
    "evaluation.answer_type":  ANSWER_TYPE,
    "evaluation.batch_size":   1,
    "dataset.annotation_path": f"/val/{EVAL_ANNOTATIONS}",
    "dataset.system_prompt":   SYSTEM_PROMPT,
    "num_gpus":                len(GPUS),
}

print(f"train on {TRAIN_ANNOTATIONS}   eval on {EVAL_ANNOTATIONS}   GPUs {GPUS}")
print(f"LoRA r={LORA_R} alpha={LORA_ALPHA} targets={LORA_TARGETS}")
print(f"{EPOCHS} epochs, batch {BATCH}, lr {LR}, {NUM_FRAMES} frames per clip")
""")

# ══════════════════════════════════════════════════════════ 4. Dataset
md(r"""
---
### 4. Dataset

The worked example is **traffic anomaly detection** on a fixed intersection camera. Two things
can go wrong in a travel lane: vehicles **collide**, or a vehicle **stalls** and blocks it.
Everything else — traffic of any density, an empty road — is the negative class.

Clips are 832x480, 24 fps, 8-33 seconds, with one label per clip.

**Domain gap.** The original clips are all clear daylight. The augmented copies re-render the
same footage in **fog**, **rain** and **night**, the conditions a deployed camera actually
sees. This is the gap the lab measures and then closes.

#### 4.1 What Is In It
""")

code(r"""
train_orig = H.load_json(TRAIN_DIR / ORIGINAL)
train_aug  = H.load_json(TRAIN_DIR / AUGMENTED)
val_orig   = H.load_json(VAL_DIR / ORIGINAL)
val_aug    = H.load_json(VAL_DIR / AUGMENTED)

TASK     = info["task"]
CLASSES  = TASK["classes"]          # the labels we want detected
NEGATIVE = TASK["negative"]         # everything else
LABEL_OF = info["label_of"]

H.dataset_table(train_orig, train_aug, val_orig, val_aug, CLASSES)

train_sources = {r["source_clip"] for r in train_aug}
val_sources   = {r["source_clip"] for r in val_aug}
H.check("no clip appears in both train and val",
        not (train_sources & val_sources),
        f"{len(train_sources)} train / {len(val_sources)} val source clips")
""")

md(r"""
Every clip is labelled with one of three answers. The model is asked one question and must
reply with a single line, which makes scoring exact:
""")

code(r"""
H.prompt_table(TASK, CLASSES, NEGATIVE)
""")

md(r"""
#### 4.2 Look at the Data

Always watch the footage before training on it. One clip per label, taken straight from the
annotations. Players show the first 10 seconds.
""")

code(r"""
from IPython.display import display

for lab in CLASSES + ["none"]:
    hit = next((r for r in train_orig if r["label"] == lab), None)
    if not hit:
        continue
    print(f"label = {lab}   ({hit['id']})")
    display(H.show_video(TRAIN_DIR / hit["video"]))
""")

md(r"""
#### 4.3 Annotation Format

Each record is one clip and one answer, in the standard *llava* conversation shape: a `video`
path relative to the split directory, and two conversation turns.
""")

code(r"""
sample = train_orig[0]
print(json.dumps(sample, indent=2))
""")

md(r"""
#### 4.4 What Augmentation Adds

The same clip under each condition. This is the domain gap made visible, and the reason the
default configuration trains on the augmented annotations.
""")

code(r"""
# Pick a source clip that exists in every condition, and show them together.
by_source = {}
for r in val_aug:
    by_source.setdefault(r["source_clip"], {})[r["variant"]] = r

conditions = ["original"] + info["conditions"]
pick = next(s for s, v in by_source.items()
            if v.get("original", {}).get("label") != "none" and len(v) == len(conditions))

print(f"source clip: {pick}   label: {by_source[pick]['original']['label']}")
paths  = [VAL_DIR / by_source[pick][c]["video"] for c in conditions if c in by_source[pick]]
labels = [c for c in conditions if c in by_source[pick]]
display(H.video_grid(paths, labels))
""")

md(r"""
#### 4.5 How the Task Is Posed — and How Else It Could Be

Everything above follows from one decision made before any data was labelled: **what
question the model is asked**. That choice fixes the annotations, the metric, and what the
model can and cannot learn. It is worth making deliberately, because it is expensive to
change later — a different task means re-labelling.

This lab poses the problem as **single-label classification with a constrained answer**:

```
question:  Which of the following does this clip show: a collision between vehicles,
           a stalled vehicle blocking a travel lane, or neither?
           Reply with exactly one line: LABEL=<collision|stalled|none>
answer:    LABEL=stalled
```

Three properties make this work. The label set is **closed**, so every clip has exactly one
correct answer. The answer is a **single token on a single line**, so scoring is exact-match
rather than a judgment call. And the classes map **one-to-one onto downstream actions** —
VSS routes a `collision` differently from a `stalled` vehicle.

That last property is the real constraint. The task exists to drive an alert pipeline, so
the label set is dictated by what the pipeline must do, not by what is easy to annotate.

##### The same problem, posed four ways

The Cosmos post-training workflow accepts two dataset families. This lab uses **`video_conversation`** —
the llava shape you saw in 4.3, free-text answers that happen to be constrained. The other
is **`task_aware_video_reasoning`**, which wraps records in a declared envelope
(`format=tao-vl-reason-v1.0`) with an explicit task name, and gets the prompt template and
answer normalization from the runtime instead of from your prompt string.

| Posing | Task name | The answer is | Scoring |
|---|---|---|---|
| **Binary question (BQ)** | `bcq` | `yes` / `no` | exact match |
| **Multiple choice (MCQ)** | `mcq` | an option letter | exact match |
| **This lab** | *(`video_conversation`)* | `LABEL=<class>` | exact match |
| **Caption** | `scene_description` | free prose | no exact match — see below |

##### What each one would change

**Binary question.** *"Does this clip show a traffic anomaly? Answer yes or no."*

- **Labels:** `collision` and `stalled` collapse into one positive class; `none` becomes the
  negative. The 128-clip validation set becomes a balanced 64/64 split.
- **Pipeline:** simpler and easier to learn — one decision boundary instead of three. The
  confusion matrix you printed in section 8 is already binary, because that is the level the
  headline metrics are computed at.
- **What you lose:** the routing. VSS could no longer tell a collision from a stalled
  vehicle, so a second stage would have to. You have moved work, not removed it.
- **Watch out:** chance accuracy is 50%, so a weak model looks far better than it is. Report
  macro-F1, never accuracy.

**Multiple choice.** *"A: collision  B: stalled vehicle  C: neither"* → answer `B`.

- **Labels:** the same three classes plus an explicit option list per record, and the answer
  becomes a letter.
- **Pipeline:** the most robust to score — a letter cannot be misspelled or hedged, and
  `mcq` participates in the workflow's deterministic accuracy automatically.
- **What you lose:** the letter carries no meaning. `B` is not "stalled"; it is just the
  second thing on a list, so the model can learn **position bias** instead of the concept.
- **Watch out:** shuffle the option order per record. If `B` is always the right answer, you
  will train a model that answers `B`.

**Caption.** *"Describe what happens in this clip."*

- **Labels:** a written description per clip. By far the most expensive to produce, and the
  only one where two annotators will disagree.
- **Pipeline:** changes the most. There is no exact match, so you need BLEU/ROUGE, embedding
  similarity, or an LLM judge — all noisier and none directly comparable to the 0.961
  macro-F1 you measured here. In the Cosmos post-training workflow, description-style tasks are **excluded from aggregate
  accuracy** with a stated reason, precisely because there is no deterministic answer.
- **What you gain:** it is the only posing that generalises past your label set. A caption
  can describe a debris spill you never anticipated; a 3-way classifier will call it `none`.
- **Watch out:** you cannot threshold prose. Something downstream still has to turn the
  description into a decision.

##### Choosing

Work backwards from the decision the output drives:

- A **closed set of actions** → classification or MCQ.
- A **single yes/no gate** → BQ.
- **Open-ended understanding**, or a label set you expect to keep extending → caption, and
  accept the weaker metric.

A useful pattern in production is to combine them: a cheap BQ gate to reject the 95% of
clips that are ordinary, then a classification or caption pass on what survives. That is
structurally what VSS already does — Behavior Analytics is the cheap gate, and the model you
just post-trained is the expensive second opinion.
""")

# ══════════════════════════════════════════════════════════ 6. Baseline eval
md(r"""
---
### 5. Baseline Evaluation

Everything heavy runs inside the Cosmos-RL container. `H.run_container(...)` assembles one
`docker run` command; the arguments are:

| argument | meaning |
|---|---|
| `job_name` | container name, so a run can be found and stopped (`docker rm -f <name>`) |
| `GPUS` | becomes `--gpus "device=0,1"` — which cards the container may use |
| `mounts` | host directory → container path. `:ro` marks read-only |
| `command` | what runs inside: `cosmos-rl-evaluate` here, `cosmos-rl ... hook.py` for training |
| `log_path` | the full container log; the notebook prints a filtered live view |

The mounts are the contract. The container only ever sees `/ptm` (the model), `/val` (the
clips and annotations) and `/results` (where it writes), so paths inside the config file are
container paths, not host paths.

The next cell prints the exact command before running anything.
""")

code(r"""
H.show_docker_command(
    "dli_eval_baseline", GPUS,
    [(PTM, "/ptm:ro"), (VAL_DIR, "/val:ro"), (EVALS / "baseline", "/results")],
    ["cosmos-rl-evaluate", "--config", "/results/config.toml"])
""")

md(r"""

Measure the untuned model on the held-out validation clips. Without this baseline, any
improvement after training cannot be quantified.

The metric is **macro-F1**: the mean of the positive-class and negative-class F1. Accuracy
alone is misleading on an imbalanced set, since a model that predicts `none` for every clip
can score well on accuracy while detecting nothing.

> This evaluation reads `EVAL_ANNOTATIONS` from section 3. Changing it there moves this run
> and the post-trained run in section 7 together, preserving the before/after comparison.

> **Expected message — the first line of output looks like a failure and is not.**
> Every container run in this notebook starts by printing
> `ERROR: This container was built for NVIDIA Driver Release 595.45 or later`.
> The word `ERROR` comes from the container entrypoint comparing its build-time driver
> against this machine's; it is a version notice. **Nothing is wrong, nothing needs
> restarting, and you do not need to do anything — keep waiting.** You will see it four
> times (here, training, the post-trained evaluation, and the merge), each followed by a
> reminder in the log stream. Judge every run by its exit code and the metrics that follow.
""")

code(r"""
def eval_config(lora_dir_in_container=None, annotations=None):
    # Start from eval_config.toml, then apply EVAL_SETTINGS and whatever this
    # particular run needs (adapter path, annotation file).
    settings = dict(EVAL_SETTINGS)
    if lora_dir_in_container:
        settings["model.model_name"] = lora_dir_in_container
        settings["model.enable_lora"] = True
        settings["model.export_dir"] = "/results/merged"
    if annotations:
        settings["dataset.annotation_path"] = f"/val/{annotations}"
    return H.apply_settings(H.load_toml(EVAL_TOML), settings)


def run_eval(name, lora_host=None, annotations=None, base=None):
    # `base` swaps the checkpoint mounted at /ptm — used to score the merged model.
    out = EVALS / name; out.mkdir(parents=True, exist_ok=True)
    cfg = eval_config("/lora" if lora_host else None, annotations)
    H.write_toml(cfg, out / "config.toml")
    mounts = [(Path(base or PTM), "/ptm:ro"), (VAL_DIR, "/val:ro"), (out, "/results")]
    if lora_host:
        mounts.append((Path(lora_host).resolve(), "/lora:ro"))
    rc, dt = H.run_container(f"dli_eval_{name}", GPUS, mounts,
                             ["cosmos-rl-evaluate", "--config", "/results/config.toml"],
                             LOGS / f"eval_{name}.log")
    items, path = H.load_eval_results(out, CLASSES)
    return H.score(items), dt

RUN_BASE_EVAL = True
val_eval_records = H.load_json(VAL_DIR / EVAL_ANNOTATIONS)

def load_or_run(name, run, label):
    # With the toggle off, fall back to a previous run's results so the later
    # comparison cells still work instead of dying on an undefined name.
    if run:
        return run()
    try:
        items, _ = H.load_eval_results(EVALS / name, CLASSES)
        print(f"reusing cached results for '{name}'")
        return H.score(items), 0.0
    except Exception:
        raise RuntimeError(f"{label} is disabled and no cached results exist "
                           f"under {EVALS / name} — enable it once.")

if RUN_BASE_EVAL:
    print(f"Evaluating the BASE model on {EVAL_ANNOTATIONS} "
          f"({len(H.load_json(VAL_DIR / EVAL_ANNOTATIONS))} clips)...\n")
base_metrics, base_time = load_or_run("baseline",
                                      (lambda: run_eval("baseline")) if RUN_BASE_EVAL else None,
                                      "RUN_BASE_EVAL")
H.metrics_table([("Cosmos 3 Nano (base)", base_metrics)],
                title="BASELINE, evaluated on " + EVAL_ANNOTATIONS,
                note="Precision 1.000 with recall 0.500 means: everything it flags is right,\n"
                     "but it misses half of what it should flag.")
""")

# ══════════════════════════════════════════════════════════ 7. Training
md(r"""
#### Where the base model fails

The single number hides *where* the model breaks down. The validation set holds the same 32
clips four times over — clean, and re-rendered as fog, rain and night — so the predictions can
be grouped by condition.

Read the per-class row above first: `collision` is already detected 32/32, while `stalled` is
0/32. The model does not have a weather problem yet; it has a **class** problem, and the table
below should confirm that by showing the same score in every condition.
""")

code(r"""
base_res = next((EVALS / "baseline").rglob("results.json"))
base_cond, base_wrong, unmatched = H.per_condition(base_res, val_eval_records, CLASSES)
H.check("every prediction matched an annotation", unmatched == 0, f"{unmatched} unmatched")
H.condition_table(base_cond, "BASE MODEL, by capture condition")
""")

md(r"""
---
### 6. Post-Training

#### 6.1 Training Configuration

Cosmos-RL is driven from a TOML spec. The keys that matter here:

| key | value | why |
|---|---|---|
| `custom.train_dataset.annotation_path` | `TRAIN_ANNOTATIONS` | the switch from section 1 |
| `policy.lora.r` / `lora_alpha` | 16 / 32 | adapter capacity; alpha/r = 2x scaling |
| `policy.lora.target_modules` | `q_proj, v_proj` | which projections get adapters |
| `policy.model_max_length` | 40960 | **do not lower** — video inputs hit a `vision_embeds` shape mismatch below ~40k |
| `train.train_policy.type` | `sft` | drop it and Cosmos-RL switches to RL mode and tries to allocate a rollout replica |
| `train.train_batch_per_replica` | 4 | must be a multiple of `mini_batch` (1) |
| `train.ckpt.max_keep` | 2 | full FSDP checkpoints are ~66 GB each. **Do not set 1** — retention then deletes the directory while the async completion-marker thread is still writing into it, and the run dies with `FileNotFoundError: .rank_0_complete`. Delete them in the cleanup cell instead. |
""")

code(r"""
RUN_TRAINING_WILL_RUN = True   # set False to keep an existing run and only re-evaluate

n_train = len(H.load_json(TRAIN_DIR / TRAIN_ANNOTATIONS))
spec = H.apply_settings(H.load_toml(TRAIN_TOML), TRAIN_SETTINGS)

RUN = RUNS / f"lora_r{TRAIN_SETTINGS['policy.lora.r']}_" \
             f"{Path(TRAIN_ANNOTATIONS).stem.split('_')[-1]}"
# Cosmos-RL writes output/<timestamp>/ per run but keeps a single output/best.
# Leaving old runs in place makes `best` ambiguous, so clear the tree first.
if RUN.exists() and RUN_TRAINING_WILL_RUN:
    shutil.rmtree(RUN)
RUN.mkdir(parents=True, exist_ok=True)
H.write_toml(spec, RUN / "config.toml")

print(f"training on   {TRAIN_ANNOTATIONS}  ({n_train} clips)")
print(f"validating on {EVAL_ANNOTATIONS}")
print(f"output        {RUN}")
print(f"steps/epoch   {n_train} samples / batch {BATCH} = {n_train // BATCH}")
""")

md(r"""
##### Choosing these values

The table above says what each key *is*. This one says what happens when you move it, which
is what you actually need when a run OOMs, diverges, or underfits.

Two framings worth holding on to. **Capacity** knobs (`r`, `target_modules`, epochs) control
how much the adapter can learn — too little and it underfits, too much and it memorises 92
clips. **Throughput** knobs (batch, `mini_batch`, frames, `dp_shard_size`) control what fits
in 190 GB of GPU memory and how fast it runs; they change the answer only indirectly.

| key | this lab | ↑ increase | ↓ decrease |
|---|---|---|---|
| `policy.lora.r` | 16 | more capacity, larger adapter, slower to converge on a small set and quicker to overfit | fewer parameters; below ~8 the adapter often cannot represent a new class at all |
| `policy.lora.lora_alpha` | 32 | scales the adapter's contribution (`alpha/r`); effectively a second learning rate | weaker adaptation — the base model's behaviour dominates |
| `policy.lora.target_modules` | `q_proj`, `v_proj` | adding `k_proj`/`o_proj`/MLP projections buys capacity for a larger behaviour change, at more memory and time | attention-only is the cheapest useful setting and is enough for a new output class |
| `train.epoch` | 3 | better fit; past the point validation loss turns up, you are memorising | faster. **Keep ≥ 2** — a 1-epoch run can leave only a broken `best` symlink after checkpoint cleanup |
| `train.optm_lr` | 1e-4 | faster movement, then `NaN` loss when too high. Fix by lowering it *and* raising `optm_warmup_epochs` | more stable, may not move at all inside a short run |
| `train.train_batch_per_replica` | 4 | steadier gradients; the documented production recommendation is 32 | noisier gradients. Two hard limits below |
| `train.train_policy.mini_batch` | 1 | fewer, larger micro-batches | **the first thing to lower when training OOMs** |
| `custom.vision.nframes` | 8 | more temporal evidence — matters for collisions, which are brief. Costs tokens quadratically-ish and can exceed `model_max_length` | cheaper and faster; too few and a short event falls between sampled frames |
| `policy.parallelism.dp_shard_size` | 2 | must equal the number of visible GPUs. Raising it spreads memory across more devices | — |
| `custom.vision.total_pixels` | default | higher resolution; **a large memory driver**, often the real cause of an OOM | cheaper; small or distant objects become unresolvable |

**Two constraints on batch size that are not suggestions.** Both produce confusing failures:

1. `train_batch_per_replica` must be **divisible by** `mini_batch` — otherwise an immediate
   `AssertionError` on every rank.
2. It must also be **≤ `total_samples / dp_shard_size`** — here 92/2 = 46. Exceed it and the
   trainer completes *zero* steps, then crashes saving a checkpoint with
   `'NoneType' object has no attribute 'state_dict'`. It reads like a checkpointing bug and
   is a batch-size bug.

> **`nframes` and `fps` are mutually exclusive.** Set exactly one. Setting both makes the
> decode backend error out and silently fall back to a path that deadlocks under
> multi-worker loading. If you switch to `fps`, delete `nframes` from the spec — and change
> `train.train_policy.dataset.name` too, or you will silently reuse the old frame cache.

##### These are not magic numbers

The values here were chosen for 92 clips and a 30-minute slot, not derived from first
principles. For comparison, an AutoML sweep over this same model on a much larger traffic
dataset landed on **r=64, alpha=2048, lr=2.6e-4, batch 32** — a far more aggressive
configuration than this one, and it beat the hand-picked run by about 6 points.

Note also that this lab's learning rate is **two orders of magnitude above** the framework
default of `1e-6`. That is defensible on a tiny dataset for a few epochs, and would be
reckless on a large one. When you move to your own data, tune rather than copy — and tune
against **task accuracy**, not validation loss. Section 6.3 shows why those two disagree.
""")

md(r"""
#### 6.2 Train

Same mechanism as evaluation, with three differences: `/train` is mounted as well, the command
is `cosmos-rl` with a **dataset hook** as its final argument, and the run writes checkpoints
into `/results`.

Progress streams below as it happens — each line is prefixed with elapsed minutes. Validation
loss is reported at the end of every epoch.
""")

code(r"""
H.show_docker_command(
    "dli_train_lora", GPUS,
    [(PTM, "/ptm:ro"), (TRAIN_DIR, "/train:ro"), (VAL_DIR, "/val:ro"), (RUN, "/results")],
    ["cosmos-rl", "--config", "/results/config.toml", "/opt/cosmos_rl/tao_sft_example.py"])
""")

code(r"""
RUN_TRAINING = RUN_TRAINING_WILL_RUN
train_time = 0.0   # defined even when training is skipped, so later cells work

if RUN_TRAINING:
    rc, train_time = H.run_container(
        "dli_train_lora", GPUS,
        [(PTM, "/ptm:ro"), (TRAIN_DIR, "/train:ro"), (VAL_DIR, "/val:ro"), (RUN, "/results")],
        ["cosmos-rl", "--config", "/results/config.toml",
         "/opt/cosmos_rl/tao_sft_example.py"],
        LOGS / "train.log")
    assert rc == 0, "training failed — see the log above"
else:
    print("Training skipped (set RUN_TRAINING = True to run it).")

adapters = sorted(RUN.rglob("adapter_model.safetensors"))
print(f"\nadapters written: {len(adapters)}")
for a in adapters:
    print(f"  {a.parent.name:<12} {a.stat().st_size/1e6:.1f} MB   {a.parent}")
""")

md(r"""
#### 6.3 Validation Loss
""")

code(r"""
# Validation loss per epoch — the training-side signal.
import re
log = LOGS / "train.log"
losses = re.findall(r"avg_loss=([0-9.]+)", log.read_text(errors="ignore")) if log.exists() else []
for i, l in enumerate(losses, 1):
    print(f"  epoch {i}: val avg_loss = {float(l):.5f}")
print("\nThe epoch with the lowest validation loss is selected as 'best'. On a dataset"
      "\nof this size, later epochs commonly overfit.")
""")

md(r"""
Validation loss is the only signal available *during* training, and it is not the thing you
care about. Loss measures how probable the correct tokens were; your metric measures whether
the final answer was right. They usually move together and they do not have to.

The gap is not a rounding error. In a published AutoML sweep over this model, the trial with
the **lowest validation loss was not the trial with the highest accuracy** — the best-loss
run scored slightly *worse* on the task than the best-accuracy run. Selecting on loss would
have shipped the wrong checkpoint.

So treat the `best` epoch as a reasonable default, not a verdict. The number that decides
whether post-training worked is the one you measure in section 7, on held-out clips, with
the metric the application actually uses.
""")

# ══════════════════════════════════════════════════════════ 8. FT eval
md(r"""
---
### 7. Post-Trained Evaluation

Same held-out clips, same annotation file, same metric — the only change is the LoRA adapter.

> Validation loss and task accuracy measure different things. A model can reduce loss by
> learning the output format alone. The task metric is therefore re-run here.
""")

code(r"""
# Pick the adapter from the epoch with the LOWEST validation loss.
# Cosmos-RL writes output/best/best_score.json for this. Note its sibling
# `best/checkpoints` symlink points at a CONTAINER path (/results/...), so it
# dangles on the host — read the JSON and map the epoch name yourself.
def best_adapter(run_dir: Path) -> Path:
    score = next(run_dir.rglob("best/best_score.json"), None)
    if score is None:
        raise FileNotFoundError(
            f"no best/best_score.json under {run_dir}; train first, or point at a run")
    meta = json.loads(score.read_text())

    # best_ckpt_abs_dir is a CONTAINER path such as
    # /results/output/<run_id>/checkpoints/epoch_3. Both the run id and the epoch
    # matter: re-training into the same directory leaves several output/<run_id>
    # trees behind, and matching on the epoch alone can silently select an
    # adapter from an earlier run.
    best_ckpt = Path(meta["best_ckpt_abs_dir"])
    epoch, run_id = best_ckpt.name, best_ckpt.parent.parent.name
    hit = run_dir / "output" / run_id / "safetensors" / epoch / "adapter_model.safetensors"
    if not hit.is_file():
        raise FileNotFoundError(
            f"best checkpoint names run {run_id}/{epoch} but {hit} is missing")

    others = {p.parent.parent.parent.name
              for p in run_dir.rglob("safetensors/*/adapter_model.safetensors")} - {run_id}
    if others:
        print(f"note: {len(others)} earlier training run(s) present in {run_dir} "
              f"({', '.join(sorted(others))}); selecting {run_id} from best_score.json")
    print(f"best epoch = {epoch} of run {run_id}  "
          f"({meta['metric']}={meta['best_score']:.5f})")
    return hit.parent

RUN_POST_EVAL = True

best = best_adapter(RUN)
if RUN_POST_EVAL:
    print(f"evaluating adapter: {best}\n")
post_metrics, post_time = load_or_run("posttrained",
                                        (lambda: run_eval("posttrained", lora_host=best)) if RUN_POST_EVAL else None,
                                        "RUN_POST_EVAL")
H.metrics_table([("Cosmos 3 Nano + LoRA", post_metrics)],
                title="POST-TRAINED, evaluated on " + EVAL_ANNOTATIONS)
""")

# ══════════════════════════════════════════════════════════ 9. Compare
md(r"""
---
### 8. Compare Before vs After

Both rows use the same clips, annotation file and metric, so the difference is attributable
to the adapter.
""")

code(r"""
H.metrics_table([("base (zero-shot)", base_metrics),
                 ("Cosmos 3 Nano + LoRA", post_metrics)],
                title="Evaluated on " + EVAL_ANNOTATIONS)
d = post_metrics["macro_f1"] - base_metrics["macro_f1"]
print(f"\nmacro-F1 change: {d:+.3f}")
""")

md(r"""
#### Why precision went *down*

One number in that table moved the wrong way: precision fell from **1.000 to 0.928**. It is
worth understanding, because it is the most commonly misread result in a post-training run.

Precision is `correct positives / all positives predicted`. Its denominator is not the
dataset — it is however many times the model chose to answer *anomaly*. The two models did
not answer the same number of times, so the two precisions are not measuring the same thing.
""")

code(r"""
H.confusion_table([("base (zero-shot)", base_metrics),
                   ("Cosmos 3 Nano + LoRA", post_metrics)])

b, t = base_metrics, post_metrics
print(f"\nmissed anomalies (false negatives):  {b['fn']:>3}  ->{t['fn']:>3}")
print(f"false alarms      (false positives):  {b['fp']:>3}  ->{t['fp']:>3}")
gained, cost = b["fn"] - t["fn"], t["fp"] - b["fp"]
if cost > 0:
    print(f"\n{gained} missed anomalies recovered for {cost} new false alarms "
          f"({gained/cost:.1f} recovered per false alarm).")
""")

md(r"""
Read the left matrix as a strategy: the base model answered *anomaly* only 32 times, and was
right all 32. That is what perfect precision buys you — and it cost 32 missed anomalies,
every single stalled vehicle in the set. A model that stays quiet is trivially precise. In
the limit, a model that never answers *anomaly* at all has undefined precision and is
completely useless.

The post-trained model answers *anomaly* far more often — the `said anomaly` row — and is
right on nearly all of them. It found **every** anomaly in the set (recall 1.000, false
negatives 0) and picked up a handful of false alarms doing it. Precision dropped because the
denominator roughly doubled, not because the model got worse at anything. The exact count
varies a little from run to run (LoRA training is not bit-reproducible across GPU types); in
the reference run on two H100s it was 5 false alarms out of 69 predictions.

Whether that trade is good is a question about **your** application, not about the metric:

| | base | post-trained |
|---|---|---|
| A collision nobody reviews | 32 missed | 0 missed |
| A clip an operator opens and dismisses | 0 | a few (see `false alarms` above) |

For alert verification these costs are wildly asymmetric — a missed collision is the failure
the system exists to prevent, while a false alarm costs a few seconds of an operator's
attention. This is why **macro-F1** is the headline metric here and precision alone is not:
macro-F1 averages the anomaly-class and none-class F1, so neither over-flagging nor
staying silent can score well on its own.

> **When precision *is* the number to protect:** if every positive triggers something
> expensive or irreversible — dispatching a vehicle, taking a system offline, contacting a
> person — then a false positive is not a few wasted seconds and you tune the other way.
> Decide which error is worse *before* you read the table.
""")

md(r"""
#### Where the post-trained model stands

Same breakdown, same clips. Two things change at once: the class gap closes, and a *weather*
gap appears where there was none before — the clean clips are now perfect while fog, rain and
night trail slightly. That residual is the specification for the next round of data.
""")

code(r"""
post_res = next((EVALS / "posttrained").rglob("results.json"))
post_cond, post_wrong, _ = H.per_condition(post_res, val_eval_records, CLASSES)

print(f"{'condition':<12}{'clips':>7}{'accuracy: base':>16}{'tuned':>9}{'change':>9}")
print("-" * 54)
for c in sorted(post_cond):
    b_acc = base_cond[c]["ok"] / base_cond[c]["n"]
    t_acc = post_cond[c]["ok"] / post_cond[c]["n"]
    print(f"{c:<12}{post_cond[c]['n']:>7}{b_acc:>16.3f}{t_acc:>9.3f}{t_acc-b_acc:>+9.3f}")
print("-" * 54)

if post_wrong:
    print(f"\n{len(post_wrong)} clip(s) still wrong:")
    print(f"  {'condition':<10}{'true label':<12}{'predicted':<12}{'clip'}")
    for cond, true_label, clip_id, pred in post_wrong[:10]:
        said = "an anomaly" if pred == "yes" else "nothing"
        print(f"  {cond:<10}{true_label:<12}{said:<12}{clip_id}")
else:
    print("\nNo clips remain wrong.")
""")

md(r"""
#### What changed, clip by clip

Aggregate numbers are easy to distrust. These are individual clips the base model got wrong
and the post-trained model gets right — one per class where possible, played below with both
answers.
""")

code(r"""
base_clips  = H.per_clip(base_res,  val_eval_records, CLASSES)
post_clips = H.per_clip(post_res, val_eval_records, CLASSES)

examples = H.find_improvements(base_clips, post_clips, limit=3)
print(f"{len(examples)} example(s) where post-training fixed the answer\n")
for cid in examples:
    H.improvement_report(cid, base_clips[cid], post_clips[cid], VAL_DIR)
    print()
""")

code(r"""
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(9, 2.8), dpi=90)
names = ["base", "LoRA"]
for i, k in enumerate(["macro_f1", "precision", "recall"]):
    vals = [base_metrics[k], post_metrics[k]]
    b = ax[i].bar(names, vals, color=["#8899a6", "#76b900"])
    ax[i].set_title(k.replace("_", "-"))
    ax[i].set_ylim(0, 1.05); ax[i].bar_label(b, fmt="%.3f", padding=3)
    ax[i].spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Cosmos 3 Nano on held-out clips — {TASK['name']}", y=1.02)
plt.tight_layout()
plt.close(fig)
fig   # rendered inline below
""")

# ══════════════════════════════════════════════════════════ 11. Merge
md(r"""
---
### 9. Merge the Adapter for Deployment

The adapter is 15 MB and loads *on top of* the base model — which is ideal for experiments,
and may not be supported by different deployment methods. **For VSS too, we need merged checkpoints.**

Folding the adapter into the weights produces a standalone checkpoint. The operation is the
LoRA definition:

$$W_{merged} = W + \frac{\alpha}{r}\,(B A)$$

where `lora_A` has shape `[r, in]` and `lora_B` has shape `[out, r]`, so `B @ A` matches the
shape of `W`. The architecture is unchanged, so the merged directory is a drop-in replacement
for the base checkpoint and carries its processor and tokenizer.

One ordering requirement: **merge before teardown**, since section 12 deletes the training
checkpoints.

The merge itself is performed by `cosmos_rl.utils.lora_utils.merge_lora_model`, the supported
implementation and the same routine the evaluator uses when it merges an adapter before
inference. `helpers/dli_helpers.py` doubles as the in-container entrypoint for it and records
provenance alongside the output.
""")

code(r"""
MERGED = Path(os.environ.get("DLI_MERGED", WORK / "merged" / "Cosmos3-Nano-VLM-lora"))
MERGED.parent.mkdir(parents=True, exist_ok=True)

# Every Cosmos-RL container is given HOME=/results and its torch caches under it. The
# merge container has no results mount of its own, so provide a writable scratch
# directory there or torch fails to create /results/.inductor.
SCRATCH = WORK / "scratch"; SCRATCH.mkdir(parents=True, exist_ok=True)

# The merge entrypoint is this lab's helper module. The Docker daemon cannot see
# the notebook folder, only the course data volume, so stage a copy next to the
# run outputs and mount that.
SCRIPTS = WORK / "scripts"; SCRIPTS.mkdir(parents=True, exist_ok=True)
shutil.copy2(HELPER_DIR / "dli_helpers.py", SCRIPTS / "dli_helpers.py")

print("merging the adapter into the base weights")
print(f"  base     {PTM}")
print(f"  adapter  {best.relative_to(WORK) if str(best).startswith(str(WORK)) else best}")
print(f"  output   {MERGED}")
print()

# The merge runs inside the Cosmos-RL container: it has torch and safetensors, and this
# keeps the notebook kernel free of heavyweight dependencies.
rc, merge_time = H.run_container(
    "dli_merge_lora", GPUS,
    [(PTM, "/ptm:ro"), (Path(best).resolve(), "/lora:ro"),
     (MERGED.parent, "/out"), (SCRATCH, "/results"), (SCRIPTS, "/scripts:ro")],
    ["python", "/scripts/dli_helpers.py", "merge",
     "--base", "/ptm", "--adapter", "/lora", "--out", f"/out/{MERGED.name}"],
    LOGS / "merge.log", stream=False)
assert rc == 0, f"merge failed - see {LOGS / 'merge.log'}"

shards  = sorted(MERGED.glob("*.safetensors"))
size_gb = sum(f.stat().st_size for f in MERGED.iterdir() if f.is_file()) / 1e9
info_m  = H.load_json(MERGED / "merge_info.json")

print("\nmerged checkpoint")
print("-" * 58)
print(f"  location      {MERGED}")
print(f"  size          {size_gb:.1f} GB across {len(shards)} shards")
print(f"  from adapter  r={info_m['r']}, alpha={info_m['lora_alpha']}, "
      f"targets {', '.join(info_m['target_modules'])}")
print(f"  contents      weights + tokenizer + chat template + preprocessor")
print("-" * 58)
print("This directory loads exactly like the base model. No adapter needed.")
""")

md(r"""
#### Confirm equivalent behaviour

The numeric check confirms the weights moved by the correct amount. This check confirms the
outcome that matters: the merged checkpoint, evaluated with no adapter loaded, scores the same
as the adapter. A disagreement between these rows indicates the merged model should not be
used.
""")

code(r"""
EVAL_MERGED = True    # ~2 extra minutes; the real proof the artifact is deployable

if EVAL_MERGED:
    merged_metrics, merged_time = run_eval("merged", base=MERGED)

    H.metrics_table([("base", base_metrics),
                     ("base + LoRA adapter", post_metrics),
                     ("merged checkpoint", merged_metrics)],
                    title="Same clips, three model forms")
    same = all(abs(merged_metrics[k] - post_metrics[k]) < 1e-9
               for k in ("macro_f1", "precision", "recall", "accuracy"))
    H.check("merged checkpoint scores identically to the adapter", same)
else:
    print("merged-checkpoint evaluation skipped (set EVAL_MERGED = True)")
""")

code(r"""
print(f"{'stage':<26}{'minutes':>9}")
print("-" * 35)
for nm, sec in [("baseline evaluation", globals().get("base_time")),
                ("LoRA training", globals().get("train_time")),
                ("post-trained evaluation", globals().get("post_time")),
                ("adapter merge", globals().get("merge_time")),
                ("merged evaluation", globals().get("merged_time"))]:
    print(f"{nm:<26}{(sec or 0)/60:>9.1f}")
""")

md(r"""
#### Handing it over

`MERGED` is a complete `qwen3_vl` checkpoint — weights, tokenizer, chat template and
preprocessor configs — and can be referenced exactly like the base model. It also contains
`merge_info.json`, recording the source adapter, the scale applied and the number of weights
modified, which makes the artifact traceable to this run.

Retain the 15 MB adapter as well. It is inexpensive to archive and can be re-merged against a
future base revision.
""")

md(r"""
---
### 10. Tear Down

Release the GPU and reclaim disk. FSDP training checkpoints are large; the adapter is not.
""")

code(r"""
subprocess.run(["docker", "rm", "-f", "dli_train_lora", "dli_eval_baseline", "dli_eval_posttrained",
                "dli_eval_merged", "dli_merge_lora"],
               capture_output=True)
ckpts = list(RUN.rglob("checkpoints"))
size = sum(f.stat().st_size for c in ckpts for f in c.rglob("*") if f.is_file()) / 1e9
print(f"training checkpoints: {size:.1f} GB in {len(ckpts)} dir(s)")
print("adapters kept:", [f"{a.stat().st_size/1e6:.0f} MB" for a in sorted(RUN.rglob('adapter_model.safetensors'))])
# Delete the large FSDP training checkpoints; the LoRA adapters are preserved.
for c in ckpts:
    shutil.rmtree(c, ignore_errors=True)
print(f"reclaimed {size:.1f} GB")
print(f"kept: adapters in {RUN}")
print(f"kept: merged checkpoint in {MERGED}")
""")

md(r"""
---
### Review

In this notebook you covered:
- **When post-training is warranted**, and how to distinguish a genuine capability ceiling
  from an inadequate prompt
- **Defining a task** as a label set and a prompt, allowing the same pipeline to be retargeted
  by editing one file
- **Reading a prepared dataset** in LLaVA conversation format, and reviewing clips before
  training
- **Splitting on source-clip identity**, so augmented re-renders of the same footage cannot
  span train and validation
- **macro-F1** as the appropriate metric for an imbalanced task
- **Establishing a baseline before training**, without which improvement cannot be claimed
- **LoRA post-training** with Cosmos-RL: adapter rank, the required specification
  keys, and why the artifact is 15 MB rather than 17 GB
- **Validation loss compared with task accuracy**: the lowest-loss epoch is not necessarily
  the most accurate checkpoint
- **Merging the adapter** into a standalone checkpoint for deployment targets that load a
  plain model
- **Gap analysis by condition**, which converts an observed weakness into a specification for
  the next data-collection run

**Next:** the same post-trained checkpoint is deployed behind VSS for alert verification, and
Part 2 of the day runs this whole workflow again through Agent Skills instead of cells.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
# ensure_ascii=False and a trailing newline match what Jupyter writes, so a
# notebook saved in the browser and one produced here diff cleanly.
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {OUT}  ({len(cells)} cells: "
      f"{sum(1 for c in cells if c['cell_type']=='code')} code / "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
