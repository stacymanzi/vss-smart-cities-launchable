# Part 2.3: Fine-Tune for Alert Verification With TAO

Lab 3 post-trained Cosmos Reason 3 Nano on traffic clips cell by cell. In this
walkthrough you run the same workflow through Agent Skills: you tell a coding
agent what to do, it reads the NVIDIA TAO Skill Bank to learn how TAO trains
and evaluates Cosmos Reason, and it launches the work in the TAO container.

You establish a zero-shot baseline on held-out clips, fine-tune with LoRA on
both GPUs, evaluate again on the same clips, and merge the adapter into a plain
checkpoint of the kind VSS serves. The next walkthrough swaps that checkpoint
into the traffic pipeline.

```{nvlearning-meta}
- **Level:** Intermediate
- **Duration:** 30-minute slot; about 27 minutes of work including approvals
- **Agent:** Codex with TAO Skill Bank 7.2.0
- **Working directory:** `/dli/task/data/lab3/part2`
- **GPUs:** 0 and 1 together, one FSDP shard each
```

## Learning Objectives

```{nvlearning-objectives}
- **Install** the TAO Skill Bank 7.2.0 into the coding agent, offline.
- **Describe** the model-level evaluation dataset and the answer contract it enforces.
- **Run** a zero-shot evaluation of Cosmos Reason 3 Nano and interpret the KPIs.
- **Verify** the training dataset and review the training plan before launching.
- **Launch** LoRA supervised fine-tuning in the TAO Cosmos-RL container.
- **Evaluate** the fine-tuned checkpoint and quantify the improvement.
- **Merge** the adapter for deployment and hand the checkpoint to Part 2.4.
```

## Before You Begin

Same VM, lab container, remote Docker daemon, and `HOST_IP` rules as Part 2.1.
Everything Part 2.1 says about namespaces applies here.

- The TAO 7.2.0 Cosmos-RL container image
  (`nvcr.io/nvidia/tao/tao-toolkit:7.2.0-cosmos-rl`, about 41 GB) is pre-pulled
  at launch. **Nothing in this walkthrough pulls an image or downloads a
  model.** No NGC key and no Hugging Face token are used.
- Course assets live under `/dli/task/data/lab3`, visible at the same path
  inside `dind`:
  - `models/Cosmos3-Nano-VLM/` — Cosmos Reason 3 Nano converted to Qwen3-VL
    safetensors, 17 GB, the training base.
  - `dataset/` — train: 92 records; val: 128 records (32 held-out clips ×
    original, fog, rain, night); `annotations_augmented.json` per split;
    `dataset_info.json` with the system prompt and question.
  - `tao-skill-bank/` — TAO Skill Bank 7.2.0.
- The Lab 3 notebook folder `/dli/task/lab-3-vlm-pt` supplies `configs/` and
  `helpers/`. This walkthrough generates its configuration files from them, so
  Part 1 and Part 2 train the same thing.
- Everything written here goes to `/dli/task/data/lab3/part2/`: `PLAN.md`,
  `jobs/<step>/config.toml` and that step's outputs, `merged/`, `scripts/`.
  **The daemon can only see paths under `/dli/task/data`.**
- The lab runs as **uid 0** by platform design (single-user container). The
  validated container runs used `-u 0:0`.

:::{important}
**Two evaluations, not one.** This walkthrough measures the *model* on Lab 3's
128 held-out clips (`LABEL=` answers, macro-F1). Parts 2.1 and 2.4 measure the
*system* on four evaluation videos (Behavior Analytics candidates, Cosmos
verdicts). The two numbers are never equated.
:::

### The Recipe

Same as Lab 3: LoRA r=16, alpha=32 on `q_proj` and `v_proj`; 3 epochs; batch 4;
learning rate 1e-4; 8 frames per clip; `model_max_length` 40960;
`annotations_augmented.json`; `dp_shard_size` 2.

Measured stage times: baseline evaluation 1.1 min, LoRA training 4.4 min,
post-trained evaluation 1.6 min, adapter merge 0.6 min, merged evaluation
1.2 min.

### The Agent Executes a Sealed Plan

`PLAN.md` and the four TOML files are generated from Lab 3's configs. The agent
runs them through the Skill Bank's launch gate (`tao-launch-workflow`), its
Docker platform skill (`tao-run-on-docker`, with job records), and uses
`tao-finetune-cosmos-reason` as the reference for what each setting means.

It does **not** run that skill's own planner (`cosmos_workflow.py`). The
planner targets a 4-GPU SLURM node — a 256 GB GPU memory floor and 384 GiB
result disk — and refuses uid 0. The skills here execute a plan that was
already validated on this VM class.

## Step 1: Stop the Part 2.1 Deployment and Free Both GPUs

In a JupyterLab terminal — the Part 2.1 shell or a new one:

```bash
cd /opt/vss && python3 experiments/aicity_track4_alert_verification/deployment/deploy.py \
  --build-dir /opt/vss/_builds/aicity-track4-alert-verification-dli-cvfix4 down
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
docker ps
```

**Expected:** no containers listed; both GPUs under 1 GB used.

:::{warning}
Never pass `-v`, and never run `docker system prune`. The warm image cache is
what makes Part 2.4's redeploy fit its slot. State volumes survive this
shutdown, so the Part 2.1 rows stay in the UI.
:::

## Step 2: Run the Notebook Setup Cells

Open `/dli/task/walkthrough-2.3-tao-skills/walkthrough_2_3_tao_skills.ipynb`
and run the **section 0** cell (GPU and container report) and the **section 1**
cell (workspace).

**Expected:** every check prints `OK` — TAO image in `dind`, base checkpoint
complete, 92 and 128 clips present, Skill Bank staged, Lab 3 configs present —
then the workspace, plan, jobs, and skill bank paths, and whether the Part 2.4
checkpoint is staged.

Section 1 writes `PLAN.md`, the four TOML files, the scoring script, and a
lab-local copy of the Skill Bank at `~/tao-skill-bank` with the marketplace
symlink restored.

## Step 3: Install the TAO Skills

The Skill Bank's own Codex installer, pointed at the local copy:

```bash
export TAO_SKILL_BANK_PATH=$HOME/tao-skill-bank
TAO_SKILL_BANK_MARKETPLACE=$TAO_SKILL_BANK_PATH bash $TAO_SKILL_BANK_PATH/scripts/install-codex-agents.sh
codex plugin list
```

**Expected:** the plugin `tao-skill-bank` is listed, and `~/.codex/AGENTS.md`
begins with `# TAO Claw Agent`.

:::{only} internal
Talking points while it runs: the bank ships 53 model skills, 10 data skills, 7
platform skills, and 13 application workflows. Codex does not run plugin
session hooks, so the identity is installed as a file and the bank path is
exported by hand.
:::

## Step 4: Start the Agent From the Workspace

A new Codex session in a new directory, on purpose — the Part 2.1 session was
started in `/opt/vss` with the VSS skills.

```bash
cd /dli/task/data/lab3/part2
TAO_SKILL_BANK_PATH=$HOME/tao-skill-bank codex
```

Leave the approval policy at its default. Every `docker run` the skill proposes
asks for approval, and the command it shows you is the lesson.

## Step 5: Give the Agent Its Context

Paste as **one message** before anything else.

```text
Context before you begin. You are the TAO Claw agent; the TAO Skill Bank 7.2.0 is
installed and TAO_SKILL_BANK_PATH points at it. Read /dli/task/data/lab3/part2/PLAN.md
now: it is the sealed, already-materialised plan for this session, generated from a
configuration that was validated end to end on this exact machine class. Treat its
TOML files as the plan of record. Do not run cosmos_workflow.py plan/preflight/
materialize and do not regenerate the TOML - the generic Cosmos-RL launch reference
assumes a 4-GPU SLURM node (256 GB GPU memory, 384 GiB disk); this is a validated
2x H100 (95 GB each) profile and the plan encodes it.

Environment facts, all verified:
1. The Docker daemon is remote (DOCKER_HOST). `docker info` describes it; do not read
   /etc/docker/daemon.json or run sysctl here - those describe this container. There is
   no sudo and none is needed.
2. Bind-mount sources are resolved by the daemon. Only /dli/task/data/... exists on both
   sides. Every path in PLAN.md is under /dli/task/data/lab3. Never mount anything else.
3. The image nvcr.io/nvidia/tao/tao-toolkit:7.2.0-cosmos-rl is already present in the
   daemon. Do not pull it and do not log in to nvcr.io. Nothing here needs NGC_KEY or
   HF_TOKEN; never ask me for a credential.
4. This lab runs as uid 0 by platform design (single-user container). The validated runs
   used -u 0:0. Apply tao-run-on-docker's documented root-required exception: run as
   uid 0, skip the post-run ownership repair.
5. Use GPUs 0 and 1 by id (--gpus "device=0,1"), exactly as the plan says. Before the
   training step, confirm both show >= 60 GB free with nvidia-smi.
6. The container prints "ERROR: This container was built for NVIDIA Driver Release
   595.45 or later" at startup. It is a version notice, not a failure.
7. Follow tao-launch-workflow's record-then-launch rule with tao_job_record.py, show me
   each docker command and a short launch review before running it, stream the log
   milestones (avg_loss, Exported safetensors, checkpoint saved), and report the exit
   code. Use the model skill's references to explain what each setting means, but the
   values come from PLAN.md.

Confirm you have read PLAN.md and summarise the five steps in one line each. Do not
launch anything yet.
```

**Expected:** a five-line summary of the plan and no commands. If it asks for
`model_type`, a credential, or a platform choice, the context did not land.
Paste it again.

## Step 6: Review the Evaluation Dataset

No GPU work. Paste into Codex:

```text
Using the dataset section of PLAN.md and the tao-finetune-cosmos-reason skill's
video_conversation dataset contract: describe the evaluation set in
/dli/task/data/lab3/dataset/val. Count records per label and per capture condition
in annotations_augmented.json, show one record verbatim, and quote the system
prompt and question from dataset_info.json. State the answer contract the model
must follow and why exact-match scoring works for it. No GPU work.
```

**Expected:** 128 records = 32 clips × original, fog, rain, night; one record
with `<video>` in the human turn and a `LABEL=` line in the answer.

The point to take away: a single `LABEL=<class>` line is what makes the metric
exact. Constrain the output format and evaluation stops being a judgment call.

## Step 7: Zero-Shot Evaluation and KPIs

Paste into Codex:

```text
Run step 1 of PLAN.md: evaluate the base checkpoint on the held-out clips with
cosmos-rl-evaluate, then score it with scripts/part2_tools.py. Report macro-F1,
precision, recall, the per-class detections, and the per-condition table. Then tell
me in two sentences what kind of gap this is.
```

**Expected (measured):** the agent opens a job record, shows the `docker run`
from `PLAN.md`, and runs about 1 minute on both GPUs.

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 0.500 |
| Macro-F1 | 0.733 |
| `collision` detected | 32/32 |
| `stalled` detected | 0/32 |

The same score appears in every condition.

:::{important}
Read that table carefully. Perfect precision with 0.500 recall, and `stalled`
at 0/32, means the base model never predicts one of the two classes. It is
identical across original, fog, rain, and night — so this is a **class gap, not
a weather gap**. Prompting will not fix a class the model never predicts.
:::

## Step 8: Review and Verify the Training Data and Plan

No GPU work. Paste into Codex:

```text
Run step 2 of PLAN.md. Verify the training split: records per label in
train/annotations_augmented.json, that no source_clip appears in both train and
val, and that every referenced video exists. Then open jobs/train_lora/config.toml
and give me the launch review the tao-launch-workflow skill asks for: image, GPUs,
data, the LoRA settings, epochs, batch, learning rate, frames per clip, and the
expected runtime from PLAN.md. Flag any key you would have set differently and why,
but do not change the file.
```

**Expected:** 92 training records (46 originals plus weather re-renders), no
train/val overlap, all clips present. The launch review lists r=16, alpha=32,
`q_proj`/`v_proj`, 3 epochs, batch 4, lr 1e-4, 8 frames, `dp_shard_size=2`,
`model_max_length=40960`.

If the agent flags `model_max_length` or `max_keep`, `PLAN.md` explains why
both stay.

## Step 9: Launch the Fine-Tune

Paste into Codex:

```text
Run step 3 of PLAN.md: launch the LoRA fine-tune on GPUs 0 and 1. Open the job
record first, then run the container and stream the milestones. When it finishes,
report the validation loss per epoch, which epoch Cosmos-RL selected as best, and
where the adapters were written.
```

**Expected:** about 4.5 minutes; three `avg_loss` lines, `Exported safetensors`,
`checkpoint saved`; adapters under
`jobs/train_lora/output/<run_id>/safetensors/epoch_N`.

:::{only} internal
While it trains, the instructor narrates: LoRA adapters on the attention
projections only, frozen base weights, one FSDP shard per GPU, validation at
the end of every epoch, a 15 MB adapter instead of a 17 GB model; the
lowest-loss epoch becomes "best".
:::

## Step 10: Evaluate the Fine-Tuned Checkpoint

Paste into Codex:

```text
Run step 4 of PLAN.md: pick the best adapter with scripts/part2_tools.py
best-adapter, evaluate it on the same clips with the same metric, and score it
against the zero-shot run with --compare. Show the before/after table and the
per-condition table, and tell me what changed and what gap remains.
```

**Expected (measured):** about 1.5 minutes.

| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| Macro-F1 | 0.733 | 0.961 | +0.228 |
| Precision | 1.000 | 0.928 | −0.072 |
| Recall | 0.500 | 1.000 | +0.500 |
| `stalled` detected | 0/32 | 32/32 | +32 |

Clean clips are perfect; fog, rain, and night trail slightly.

The class gap closed — and a small weather gap appeared where there wasn't one
before. That residual is the specification for the next data run, which is
exactly what Part 2.2 generates. This is the loop closing on itself.

## Step 11: Merge for Deployment and Prove It

Paste into Codex:

```text
Run step 5 of PLAN.md: merge the best adapter into the base weights with the
provided entrypoint, then evaluate the merged checkpoint on the same clips and
confirm it scores identically to the adapter run. Report the merged checkpoint's
path, size, shard count and the contents of merge_info.json.
```

**Expected:** about 1 minute to merge and 1.5 to evaluate;
`/dli/task/data/lab3/part2/merged/Cosmos3-Nano-VLM-lora`, about 17 GB,
identical metrics.

Why merge at all: VSS's RT-VLM loads a plain Hugging Face checkpoint, not a
base-plus-adapter pair. This is the same artifact, same recipe, and same
basename as the course-provided one that Part 2.4 deploys.

## Step 12: Verify in the Notebook

Run the **section 5** cells. This is independent of what the agent reported —
the scoring is Lab 3's.

**Expected:** a three-row table (base, LoRA adapter, merged) with the merged row
identical to the adapter row; per-condition tables for base and fine-tuned; and
a hand-off cell confirming the merged checkpoint is complete (config, every
shard, tokenizer, preprocessor, `merge_info.json`) and printing the two flags
Part 2.4 uses, with the course-provided copy reported as present.

## Step 13: Hand the Machine to Part 2.4

The TAO identity would otherwise load into the Part 2.4 Codex session too.

```bash
rm -f ~/.codex/AGENTS.md
```

Then `/exit` the TAO Codex session and continue Part 2.4 from the Part 2.1
terminal, whose shell still holds the `AICITY_*` variables and the NGC login.
Both GPUs are free and the AI City deployment is stopped with its state volumes
intact, so Part 2.4's "stop the current deployment" step finds it already
stopped.

:::{note}
Part 2.4 deploys the course-provided checkpoint at
`/dli/task/data/models/Cosmos3-Nano-VLM-lora`. Your own merged output at
`/dli/task/data/lab3/part2/merged/Cosmos3-Nano-VLM-lora` is the same recipe
with the same basename, and is a drop-in for `--vlm-model-dir` and
`--vlm-model Cosmos3-Nano-VLM-lora`.
:::

```{nvlearning-checkpoint} Checkpoint 3
- Part 2.1 deployment stopped without `-v`; both GPUs free.
- TAO Skill Bank 7.2.0 installed into Codex offline; agent confirmed the sealed plan.
- Zero-shot baseline measured: precision 1.000, recall 0.500, `stalled` 0/32, same in every condition.
- Training data verified and plan reviewed; LoRA fine-tune completed on both GPUs.
- Fine-tuned model evaluated on the same clips: recall 1.000, macro-F1 up by about 0.23.
- Adapter merged; merged checkpoint scores identically; hand-off flags for Part 2.4 printed; TAO identity removed.
```

## Troubleshooting

| It reports | What is actually true |
|---|---|
| Section 1 cell: "Something is missing" | The course loader is still running or the S3 copy is incomplete. Wait and re-run. Never download anything. |
| `install-codex-agents.sh` fails, or `codex plugin` is not a command | The lab's Codex version lacks the plugin marketplace commands. Record the version; the install path in Step 3 depends on it. |
| Agent asks for `model_type`, a platform, `NGC_KEY` or `HF_TOKEN` | The context block did not land. Paste it again; every answer is in `PLAN.md`. |
| Agent refuses to run the container as uid 0 | Point it to item 4 of the context: the platform runs the lab as root by design and the validated runs used `-u 0:0`. |
| Agent wants to run `cosmos_workflow.py plan`, or a preflight fails on 256 GB GPU memory or 384 GiB disk | Those gates describe a 4-GPU SLURM node. `PLAN.md` is the materialised plan for this VM; do not re-plan. |
| "ERROR: This container was built for NVIDIA Driver Release 595.45 or later" | A version notice printed at startup, not a failure. Judge the run by its exit code and outputs. |
| A GPU has less than 60 GB free, or training OOMs | Something is still resident: the Part 2.1 verifier, an EVG container from Part 2.2, or a Lab 2 kernel. `docker ps` and `nvidia-smi` show it; remove it and rerun. |
| `best-adapter`: no `best/best_score.json` | Training did not finish. Read `jobs/train_lora`'s log. Note that `output/best/checkpoints` is a symlink to a container path and dangles on the host by design; the script does not follow it. |
| Training dies with `FileNotFoundError: .rank_0_complete` | `train.ckpt.max_keep` was lowered to 1. It must stay 2. |
| Merged checkpoint scores differently from the adapter | Do not hand it over. Rerun the merge and compare `merge_info.json` with the adapter's config. |
| In Part 2.4 the agent asks TAO-style intake questions | `~/.codex/AGENTS.md` is still present. Remove it (Step 13) and restart Codex. |

:::{only} internal
**Shot list.**

- Section 1 cell output: all OK lines and the workspace paths.
- `codex plugin list` showing `tao-skill-bank`.
- The agent's five-line plan summary after the context block.
- The agent's launch review before the training `docker run` (image, GPUs, LoRA settings).
- Zero-shot score table: precision 1.000, recall 0.500, `stalled` 0/32.
- Training log milestones: three `avg_loss` lines and "Exported safetensors".
- Before/after score table: 0.733 to 0.961, +0.228; `stalled` 32/32.
- Per-condition table: clean perfect, fog, rain and night slightly lower.
- Merge summary: 17 GB, shard count, `merge_info.json`; "scores identically to the adapter".

**Time budget.**

30-minute slot with approvals: install 3, review 3, zero-shot 4, verify 3,
train 7, evaluate 4, merge 3 = 27 min.
:::

## What's Next

You have a merged checkpoint that scores 0.961 macro-F1 on held-out clips. That
is a model-level result. In
[Part 2.4: Deploy Fine-Tuned VSS for Alert Verification](part-2-4-deploy-fine-tuned-vss)
you put it back into the pipeline and find out what it's worth at the system
level.
