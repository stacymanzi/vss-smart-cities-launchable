# Part 2.4: Deploy Fine-Tuned VSS for Alert Verification

In Part 2.1 the tuned profile turned the stock deployment into a working
traffic system, with the stock Cosmos Reason 3 Nano confirming most, but not
all, of the real events. Here you keep everything else fixed and swap in the
checkpoint fine-tuned on the traffic dataset in Part 2.3, then
compare the two models on the same evaluation.

Holding everything else constant is the whole point. The CV stages are
untouched, so Behavior Analytics produces identical candidates. Any difference
you measure comes from the verifier and nothing else.

```{nvlearning-meta}
- **Level:** All levels — no prior VLM or Cosmos experience needed
- **Duration:** About 20 minutes of agent time
- **Agent:** Codex, continuing from the Part 2.1 terminal
- **Checkpoint:** `/dli/task/data/models/Cosmos3-Nano-VLM-lora`
```

## Learning Objectives

```{nvlearning-objectives}
- **Swap** the stock Cosmos Reason 3 Nano for the fine-tuned checkpoint in the tuned profile.
- **Run** the same evaluation video through the redeployed pipeline.
- **Compare** stock and fine-tuned results on the official four-video evaluation.
- **Interpret** why the system-level gain differs from the model-level gain.
```

## Before You Begin

Same VM, lab container, remote Docker daemon, and `HOST_IP` rules as Part 2.1.

The fine-tuned checkpoint is a **merged LoRA** (r=16, alpha=32 on `q_proj` and
`v_proj`) of Cosmos Reason 3 Nano. RT-VLM serves it from the course data
directory. Swapping models is one build flag.

Measured agent time: about 5 minutes to rebuild, about 5 minutes for RT-VLM to
load the checkpoint, about 6 minutes per 3-minute evaluation video.

:::{note}
`/dli/task/data/models/Cosmos3-Nano-VLM-lora` is either the checkpoint you merged
in Part 2.3 Step 7 or, if one was already staged there, the course-provided one.
Both are the same recipe with the same basename; everything below works
unchanged.
:::

## Step 1: Rebuild With the Fine-Tuned Checkpoint

Paste into Codex. The shell from Part 2.1 Step 8 still has the environment.

```text
Rebuild the AI City tuned profile using the README's optional --vlm-model-dir
section with /dli/task/data/models/Cosmos3-Nano-VLM-lora, keeping all other
settings as before and backing up the existing build directory first. Stop
the current deployment without -v, deploy the new build and wait for ready.
RT-VLM must advertise Cosmos3-Nano-VLM-lora.
```

**Expected:** `build.py` prints `RT-VLM serves fine-tuned checkpoint ... as
VLM_NAME=Cosmos3-Nano-VLM-lora`; readiness confirms the advertised model. About
10 minutes.

## Step 2: Open the UI

Browse to `http://<your-lab-host>/alerts/`.

Rows from Part 2.1 remain — the state volumes survived every shutdown. New rows
arrive under the new sensor id.

## Step 3: Run the Same Evaluation Video

Paste into Codex:

```text
Run README section 8 with
/dli/task/data/traffic_anomaly_dataset/eval_videos/eval_video_01.mp4,
run id ft-eval-01 under /opt/aicity-runtime/aicity-eval,
--vlm-model Cosmos3-Nano-VLM-lora, keeping --retain-vios-storage.
Show me the run manifest counts and the Cosmos verdicts, and compare them
with the stock-eval-01 run.
```

**Expected in about 6 minutes:** identical detection frames and candidates to
`stock-eval-01`, a Cosmos verdict per candidate, and the agent listing any
verdict that differs. In the UI, filter by sensor `ft-eval-01`.

The identical candidate count is the control in this experiment. If it moved,
something other than the model changed.

## Step 4: Compare on the Official Evaluation

Course-provided results from the same pipeline on all four evaluation videos —
16 events total: 8 collisions, 8 stalled vehicles.

| Metric | Stock Cosmos Reason 3 Nano | Fine-tuned |
|---|---|---|
| Behavior Analytics coverage | 16/16 | 16/16 |
| Cosmos confirmed | 14/16 | 15/16 |
| False negatives | 2 | 1 |
| Confirmed false positives | 0 | 0 |
| Changed verdict | none | `eval_video_04` event 01 (collision) becomes confirmed |

**One missed collision recovered, no regressions.**

:::{important}
Compare this with Part 2.3, where recall went from 0.500 to 1.000 on the
`LABEL=` task. Here the same weights confirm one more event out of 16. Both
numbers are correct, and they differ on purpose: Alert Bridge asks its own
yes/no question with its own frame sampling, so a model-level jump does not
transfer one-for-one to the system level. Always state which evaluation you
mean.
:::

The scoring command that produced the table, for reference — the agent runs it:

```bash
score_cosmos_super_eval.py --timelines /dli/task/data/traffic_anomaly_dataset/eval_videos/timelines.json \
  --run-dir <one per eval video> --expected-vlm-model Cosmos3-Nano-VLM-lora \
  --output /opt/aicity-runtime/aicity-eval/ft-score.json
```

## Step 5: Clean Up

Paste into Codex:

```text
Stop the AI City deployment with deploy.py down (no -v) and confirm no
containers remain under DOCKER_HOST.
```

```{nvlearning-checkpoint} Checkpoint 4
- Tuned profile redeployed with the fine-tuned checkpoint; RT-VLM advertised `Cosmos3-Nano-VLM-lora`.
- `eval_video_01` processed; candidates identical to the stock run; verdicts compared.
- Stock versus fine-tuned understood: same coverage, one more confirmed collision, no false positives.
- Deployment stopped without `-v`.
```

## Troubleshooting

| It reports | What is actually true |
|---|---|
| `install.sh`: "installed experiment differs" | An older copy of the package is installed. Move `/opt/vss/experiments/aicity_track4_alert_verification` aside and rerun. |
| `vss-rtvi-vlm` restarting; log says `Read-only file system: .../.lock` | The model directory is not writable. The package stages a writable symlink directory for this; if you see this, the package is stale. |
| Readiness: "did not advertise exact VLM_NAME" | The model id is the checkpoint directory name. Use `Cosmos3-Nano-VLM-lora` as `--vlm-model`. |
| Compose reports a dependency "unhealthy" seconds after start | First-boot Postgres race on a fresh volume. Rerun `up`. |

:::{only} internal
**Shot list.**

- `build.py` line "RT-VLM serves fine-tuned checkpoint ... VLM_NAME=Cosmos3-Nano-VLM-lora".
- Codex summary "Fine-tuned deployment is ready; RT-VLM advertises Cosmos3-Nano-VLM-lora".
- Alerts UI filtered to `ft-eval-01`.
- Codex's side-by-side comparison of `stock-eval-01` and `ft-eval-01` verdicts.
- The stock versus fine-tuned score table.
:::

## What's Next

You've closed the loop: measured, generated, fine-tuned, redeployed, and
measured again. The [Conclusion](conclusion) recaps what you built and where
to take it next.
