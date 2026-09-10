# Part 2.3: Fine-Tune for Alert Verification With TAO

Part 2.1 left you with a working pipeline whose verifier — the stock Cosmos
Reason 3 Nano — confirms most real events but not all of them. Part 2.2 gave you
the data to fix that. Here you fine-tune the verifier with LoRA, and you do it
without writing a single container command: you tell a coding agent what you
want, and the [NVIDIA TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank)
teaches it how TAO trains, evaluates and merges Cosmos Reason.

You measure the base model on held-out clips, fine-tune, measure again on the
same clips, and merge the adapter into the plain checkpoint that Part 2.4
deploys.

```{nvlearning-meta}
- **Level:** Intermediate
- **Duration:** About 30 minutes, dominated by one training wait
- **Agent:** Codex with the TAO Skill Bank
- **Working directory:** `/dli/task/data/lab3/part2`
```

## Learning Objectives

```{nvlearning-objectives}
- **Install** the TAO Skill Bank into a coding agent.
- **Run** a zero-shot evaluation of Cosmos Reason 3 Nano and read what kind of gap it shows.
- **Review** the training plan the agent proposes before approving the launch.
- **Fine-tune** Cosmos Reason 3 Nano with LoRA and evaluate the result on the same clips.
- **Merge** the adapter into a checkpoint that VSS can serve.
```

## Before You Begin

The lab environment is already loaded: the TAO container image, the base
checkpoint, the dataset and the Skill Bank are all in place, and the prompts
below assume that. On a fresh machine none of this is a blocker — given an NGC
key and a Hugging Face token, the coding agent with the TAO skills pulls the
image, downloads the model and prepares the checkpoint itself. The only
difference is time.

| Asset | Where |
|---|---|
| TAO Cosmos-RL container | `nvcr.io/nvidia/tao/tao-toolkit:7.2.0-cosmos-rl`, present in the Docker daemon |
| Base checkpoint | `/dli/task/data/lab3/models/Cosmos3-Nano-VLM/` (17 GB) |
| Dataset | `/dli/task/data/lab3/dataset/` — `train/` 92 clips, `val/` 128 clips, `dataset_info.json` |
| Course-provided fine-tuned checkpoint | `/dli/task/data/models/Cosmos3-Nano-VLM-lora/` — the fallback Part 2.4 deploys if you skip this walkthrough |

The base checkpoint is Cosmos Reason 3 Nano already converted to Qwen3-VL
safetensors, so no conversion runs here. When that converted checkpoint is not
present, the agent downloads the omni-modal `nvidia/Cosmos3-Nano` from Hugging
Face and converts it to Qwen3-VL safetensors itself before training; the skill
owns that step.

Everything under `/dli/task/data` is visible at the same path from your terminal
and from the Docker daemon (which runs in a separate `dind` container, reached
through `DOCKER_HOST`). That is the one path the daemon can bind-mount, so your
workspace lives there too.

### The Recipe

The configuration used for every number on this page. Step 5 gives it to the
agent explicitly.

| Setting | Value |
|---|---|
| Method | LoRA, supervised fine-tuning (SFT) |
| Adapter targets | `q_proj`, `v_proj` |
| LoRA rank / alpha | 16 / 32 |
| Epochs | 3 |
| Batch size | 4 |
| Learning rate | 1e-4 |
| Frames sampled per clip | 8 |
| Training annotations | `train/annotations_augmented.json` — originals plus fog, rain and night re-renders |
| GPUs | 0 and 1, one FSDP shard each |

### Expected Times

Container run times; the agent adds a minute or two of planning around each.

| Stage | 2 × H100 | 1 × RTX PRO 6000 (96 GB) |
|---|---|---|
| Zero-shot evaluation | ~1 min | ~0.7 min |
| LoRA fine-tuning | ~4.5 min | ~6.5 min |
| Fine-tuned evaluation | ~1.5 min | ~1.2 min |
| Merge | ~1 min | ~0.7 min |
| Merged-checkpoint evaluation | ~1 min | ~1 min |

### Three Things Worth Knowing Before You Prompt

The Skill Bank encodes production defaults, and this lab sits outside three of
them. The prompts below handle each; this is why.

**The patch embedding.** Cosmos-RL's default Conv3D video patch embedding has no
cuDNN engine on this machine's driver and fails on the first clip with
`GET was unable to find an engine to execute this computation`. The `linear`
implementation is numerically equivalent and works everywhere, so the prompts
ask for it. Left to discover this on its own, the agent does get there, but only after
several retries.

**GPU memory and disk.** The Cosmos-RL skill requires at least 256 GB of GPU
memory in total and 384 GiB of free disk before it will launch SFT — sized for
fine-tuning an 8B model on a full dataset, where 8 × H100 is the
recommendation. This machine has two H100s, about 190 GB. The job here is 92
clips for three epochs and fits comfortably; the prompts say so, and the agent
proceeds. The disk figure deserves respect even so: each full training
checkpoint Cosmos-RL writes is about **66 GB**, and it retains two. A run that
fills the disk fails while saving a checkpoint, with the rather opaque message
`basic_ios::clear: iostream error`. Step 9 removes these checkpoints once the
adapter has been merged.

**Running as root.** The TAO Docker skill refuses to run a container as `root`
when it writes to a bind mount. The reason is practical: a container running as
root writes root-owned files onto your disk, and afterwards you — a normal user
— cannot delete or modify them. The skill's fix is to run the container as your
own user id instead. In this lab that protection is not needed: the lab itself runs as
root in a single-user container, so there is no other user whose files could be
affected. The prompts tell the agent this, and it runs the container as root
without the extra ownership step.

## Step 1: Free the GPUs

Part 2.2 left model endpoints running — Nemotron and, if you did Track A,
Cosmos. Training needs both GPUs empty. In a JupyterLab terminal:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

Stop whatever is listed:

```bash
docker stop $(docker ps -q)
```

Rerun `nvidia-smi`. **Expected:** both GPUs under 1 GB used.

:::{warning}
`docker stop` is sufficient. Please avoid `-v` on a compose down and
`docker system prune`: the image cache is what allows every later step to fit
its time slot.
:::

## Step 2: Install the TAO Skill Bank

The Skill Bank ships a one-line Codex installer. It registers the skills as a
plugin and installs the TAO agent identity so every Codex session knows to use
them.

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA-TAO/tao-skill-bank/main/scripts/install-codex-agents.sh | bash
codex plugin list
```

**Expected:** `tao-skill-bank` listed as installed and enabled.

## Step 3: Start the Agent

A new Codex session in a new directory. The Part 2.1 session lives in
`/opt/vss` with the VSS skills; this one gets its own workspace.

```bash
mkdir -p /dli/task/data/lab3/part2 && cd /dli/task/data/lab3/part2
codex
```

Leave the approval policy at its default. Every `docker run` the agent proposes
stops for your approval, and the command it shows you is worth reading.

## Step 4: Zero-Shot Evaluation

Paste into Codex:

```text
Using the TAO skills, evaluate the base Cosmos 3 Nano checkpoint at
/dli/task/data/lab3/models/Cosmos3-Nano-VLM on the validation split in
/dli/task/data/lab3/dataset/val.

Use the Cosmos-RL backend (cosmos-rl-evaluate) with qwen3_vl_patch_embed set to
linear, 8 frames per clip, model_max_length 40960, and the system prompt from
dataset_info.json.

Score each answer by the class it names - collision, stalled or none - and treat
collision and stalled together as the positive class. Report macro-F1 over positive
vs none, precision, recall, clips detected per class, and accuracy per capture
condition.

Then tell me in two sentences what kind of gap this is.
```

The second paragraph is the only part of that prompt that is not plain intent.
The agent can reach the same result without it — on this driver the default
Conv3D patch embedding fails with `GET was unable to find an engine`, and the
agent needs a few iterations of probing and retrying before it settles on the
`linear` setting. Stating it up front saves that time; leaving it out is a
reasonable way to observe how the agent troubleshoots.

The agent will ask a few questions before it launches; the answers are short:

| It asks | Answer |
|---|---|
| Which checkpoint format, `qwen3_vl` or `cosmos3_omni`? | `qwen3_vl` — it is already converted. |
| For an NGC key or Hugging Face token | None needed. The image is present in the daemon and nothing downloads. |
| To run the container as a non-root user | The lab itself runs as root; run it as root. |
| Where to write | The current directory. The daemon can only mount paths under `/dli/task/data`, which is why you are here. |

**Expected (measured):** the agent shows you the `docker run`, waits for your
approval, and evaluates for about a minute.

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 0.500 |
| Macro-F1 | 0.733 |
| `collision` detected | 32/32 |
| `stalled` detected | 0/32 |

The same score appears in every condition. Look at the raw answers too: the base
model drops the `LABEL=` prefix on most of them and just says `none` or
`collision`. Fine-tuning fixes the format along with the content.

:::{important}
Perfect precision with recall 0.500 and `stalled` at 0/32 means the base model
never predicts one of the two classes. Precision 1.000 is not a sign of strength
in this case; it is the score a model earns by answering only the cases it is
already confident about. Because the result is identical across original, fog,
rain and night, this is a **class gap rather than a weather gap** — and a class
the model never predicts is something training addresses far more effectively
than prompting.
:::

## Step 5: Review the Training Plan

Paste into Codex:

```text
Plan a LoRA fine-tuning run of the same checkpoint on the training split in
/dli/task/data/lab3/dataset/train, validating on the val split, on both GPUs.

Use LoRA rank 16, alpha 32 on q_proj and v_proj; 3 epochs; batch size 4;
learning rate 1e-4; 8 frames per clip; one FSDP shard per GPU; and the same
Cosmos-RL backend settings as the evaluation (qwen3_vl_patch_embed linear,
model_max_length 40960).

This machine is below the skill's 256 GB GPU-memory and 384 GiB disk gates for
SFT. The job is 92 clips for three epochs and has been validated here, so
proceed past both gates.

Before launching anything, verify the training split - records per label, no clip
in both train and val, every video present - and show me the launch review with a
one-line reason for each setting.
```

**Expected:** 92 training records — 40 collision, 36 stalled, 16 none — from 46
source clips, no overlap with val, all videos present. Then a settings table
matching the recipe above, with the agent's reasoning for each row.

The recipe is spelled out because it is known to work on this dataset in this
time slot. You can instead leave the second paragraph out and ask the agent to
choose the LoRA and optimizer settings itself, explaining each choice; that is
the better first move on a dataset you have not trained on before. Expect it to
take longer to converge on a good configuration, though — in our test the
agent's own first choice (batch 1, a higher learning rate, adapters on every
projection) trained unstably and had to be replaced with the recipe above.

Two of the checks it runs are worth understanding, because both produce
misleading errors when violated:

- Batch size must be **divisible by** the micro-batch. Otherwise: an immediate
  `AssertionError` on every rank.
- Batch size must be **no larger than** `samples ÷ GPUs` — here 92 ÷ 2 = 46.
  Otherwise the run completes *zero* training steps and then crashes saving a
  checkpoint with `'NoneType' object has no attribute 'state_dict'` — an error
  that appears to concern checkpointing but is caused by the batch size.

:::{tip}
Ask the agent what it would change. The learning rate is two orders of
magnitude above the skill's default of `1e-6`, and a well-briefed agent will
point that out. The value is appropriate for 92 clips over a few epochs; on a
large dataset it would need to be tuned, which is the subject of "Beyond This
Lab".
:::

## Step 6: Launch

Paste into Codex:

```text
Launch it. Stream the milestones as it runs - the validation loss at each epoch,
checkpoint saves, and the safetensors export.

When it finishes, tell me which epoch Cosmos-RL selected as best and where the
adapter was written.
```

**Expected:** about 4.5 minutes; three validation-loss lines, one per epoch;
`epoch_3` selected as best; the adapter under `.../safetensors/epoch_3/`.

:::{only} internal
While it trains, the instructor narrates: adapters on the attention projections
only, frozen base weights, one FSDP shard per GPU, validation at the end of every
epoch, a 15 MB adapter instead of a 17 GB model. Note that only epochs 2 and 3
remain on disk — checkpoint retention keeps two.
:::

## Step 7: Evaluate the Fine-Tuned Model

Paste into Codex:

```text
Evaluate the best adapter on the same validation clips, scored exactly as the
zero-shot run was (class word with or without the LABEL= prefix; anomaly vs none).

Show me before and after side by side: the metrics, how many clips of each class
were detected, and accuracy per condition. List every clip that is still wrong.

Then tell me what changed and what gap remains.
```

**Expected (measured):** about 1.5 minutes.

| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| Macro-F1 | 0.733 | 0.961 | +0.228 |
| Precision | 1.000 | 0.928 | −0.072 |
| Recall | 0.500 | 1.000 | +0.500 |
| `stalled` detected | 0/32 | 32/32 | +32 |

Clean clips are perfect; fog, rain and night trail slightly. Every remaining
error is a *normal* clip in bad weather that the model now flags as an anomaly.

:::{note}
**Precision fell and the model got better.** The base model made 32 predictions
and got all 32 right. The fine-tuned model makes about 69 and gets 64 right:
it recovered every one of the 32 missed anomalies and picked up a few false
alarms doing it. Precision dropped because the denominator doubled, not because
the model became worse at anything. Whether the trade is worthwhile depends on
the application; for alert verification, where a miss is an unreviewed collision
and a false alarm costs an operator a few seconds, recovering 32 misses for a
handful of false alarms is a clear improvement.
:::

The class gap is closed, and a small weather gap has appeared where there was
none before. That residual is the specification for the next data run — exactly
what Part 2.2 generates. This is the loop closing on itself.

## Step 8: Merge for Deployment

VSS serves a standard Hugging Face checkpoint rather than a base model with a
separate adapter, and the Cosmos 3 Reasoner NIM has the same requirement.
Merging the adapter into the base weights is therefore the standard final step
of a Cosmos fine-tuning run.

Part 2.4 deploys whatever is at `/dli/task/data/models/Cosmos3-Nano-VLM-lora`.

Paste into Codex:

```text
Merge the best adapter into the base weights and write the merged checkpoint to
/dli/task/data/models/Cosmos3-Nano-VLM-lora. If a checkpoint already exists there,
leave it in place and skip the merge.

Evaluate the checkpoint at that path on the same clips and confirm it scores
identically to the adapter run.

Report its size and shard count.
```

**Expected:** about 1 minute to merge, 1.5 to evaluate; 17 GB in 4 shards;
metrics identical to Step 7.

## Step 9: Clean Up

Training left two full checkpoints of about 66 GB each under the run directory.
The merged checkpoint and the small adapter are all that later steps need.

```text
Delete the full training checkpoints under the run directory, keeping the
adapter safetensors, the evaluation results and the merged checkpoint.

Show me the disk space before and after.
```

```{nvlearning-checkpoint} Checkpoint 3
- Part 2.2 endpoints stopped; both GPUs free.
- TAO Skill Bank installed into Codex.
- Zero-shot baseline measured: precision 1.000, recall 0.500, `stalled` 0/32, the same in every condition.
- Training split verified and the launch plan reviewed before approval.
- LoRA fine-tuning completed; evaluated on the same clips: recall 1.000, macro-F1 up by about 0.23.
- You can explain why precision fell while the model improved.
- The checkpoint at the path Part 2.4 deploys scores identically to the adapter.
- Full training checkpoints removed; adapter, results and merged checkpoint kept.
```

## Troubleshooting

| It reports | What is actually true |
|---|---|
| `codex plugin list` does not show `tao-skill-bank` | The installer needs network access to GitHub. Rerun it; if the lab's Codex lacks `codex plugin`, ask a TA. |
| Agent asks for `NGC_KEY`, `HF_TOKEN`, or which `model_type` the checkpoint is | Expected — see the answer table under Step 4: no credentials, checkpoint is `qwen3_vl`. |
| Agent stops on the 256 GB GPU-memory or 384 GiB disk gate | Expected. Tell it the job is validated on this machine class and to proceed; see "Three Things Worth Knowing Before You Prompt". |
| Agent refuses to run the container as root | Tell it the lab itself runs as root, so there is no ownership mismatch to protect against. |
| `ERROR: This container was built for NVIDIA Driver Release 595.45 or later` | A version notice printed at container start, not a failure. Judge the run by its exit code and outputs. |
| Container exits 1 immediately, before loading anything | Its working directory is not writable. The agent should set it to the results mount (`-w /results`). |
| A GPU has less than 60 GB free, or training OOMs | Something from Part 2.2 or 2.1 is still running. `docker ps` shows it; `docker stop` it. |
| Training crashes saving a checkpoint with `'NoneType' object has no attribute 'state_dict'` | Zero training steps ran — batch size exceeded samples per GPU. See Step 5. |
| Training exits at once with `ModuleNotFoundError: No module named 'cosmos_rl'` | The training hook was run as a script. It is the final positional argument to `cosmos-rl --config <spec> <hook.py>`, not a program of its own. The agent usually catches this itself and retries. |
| Training fails while saving a checkpoint with `RuntimeError: basic_ios::clear: iostream error` | The disk is full. Each full checkpoint is ~66 GB and two are kept. Free space (old runs' `checkpoints/` trees are the usual culprit; the small `safetensors/` adapters are what you need) and rerun. |
| Training fails with `FileNotFoundError: .rank_0_complete` | Checkpoint retention was set to 1. It must stay at 2. |
| The `best` link points at a `step_N` directory that does not exist | Known Cosmos-RL behaviour: epoch-based saves write `epoch_N` while `best` records a step. The agent resolves it to the retained epoch folder. |
| Merged checkpoint scores differently from the adapter | Do not deploy it. Have the agent rerun the merge and compare `merge_info.json` with the adapter's config. |

## Beyond This Lab

### One Configuration Is Not a Tuned Configuration

You ran one LoRA configuration and gained +0.228 macro-F1. That is a large gain
from a single run, and it is unlikely to be the best available. The NVIDIA
reference workflow for this model follows the fine-tuning prompt with a second
one that hands hyperparameter search to
[TAO AutoML](https://docs.nvidia.com/tao/tao-toolkit/latest/text/automl/automl.html):

```text
Run an AutoML sweep to improve the LoRA result. Let TAO choose suitable search
strategies and tune the important training hyperparameters.

Optimize validation accuracy and summarize the best models.
```

On the Woven Traffic Safety benchmark, one LoRA run took accuracy from 54% to
87%, and a 43-trial AutoML sweep reached 93% — in about 19.5 hours and 170
GPU-hours, which is why it is out of scope for a 30-minute slot. One finding
from that sweep is more important than the headline number: the trial with the
**lowest validation loss was not the one with the highest accuracy**. Optimize
the metric you care about, and have the agent report both.

### Why LoRA and Not Full Fine-Tuning

The same sweep was run with full-parameter SFT. The best SFT trial reached the
same 93% as the best LoRA trial, at roughly seven times the GPU-hours per trial.
On this class of task LoRA matches full fine-tuning at a fraction of the cost,
which is why it is the default here and in the reference workflow.

### Other Platforms

The lab is a single Docker daemon, so the agent used the Docker platform skill.
The same model skill runs unchanged on SLURM, Kubernetes and Brev through their
platform skills; the prompts in this walkthrough work as written on any of them.

:::{only} internal
**Shot list.**

- `codex plugin list` showing `tao-skill-bank`.
- The agent's launch review before the training `docker run`.
- Zero-shot score table: precision 1.000, recall 0.500, `stalled` 0/32.
- Training log: three validation-loss lines and the safetensors export.
- Before/after score table: 0.733 to 0.961; `stalled` 32/32.
- Per-condition table: clean perfect, fog, rain and night slightly lower.
- Merge summary: 17 GB, 4 shards, "scores identically to the adapter".

**Time budget.** Free GPUs 2, install 2, zero-shot 4, plan review 4, train 7,
evaluate 4, merge 4, clean up 1 = 28 min.
:::

## What's Next

You have a merged checkpoint scoring 0.961 macro-F1 on held-out clips. That is a
model-level number. In
[Part 2.4: Deploy Fine-Tuned VSS for Alert Verification](part-2-4-deploy-fine-tuned-vss)
you put it back into the pipeline and find out what it is worth at the system
level.
