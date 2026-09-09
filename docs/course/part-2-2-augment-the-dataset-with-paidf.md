# Part 2.2: Augment the Dataset With PAIDF

Part 2.1 left you with a working pipeline and a gap: the verifier misses events
it hasn't seen enough of. Collisions and stalled vehicles are, fortunately,
rare — which is exactly what makes them hard to collect training data for.

The NVIDIA Physical AI Data Factory closes that gap two ways. You can
*generate* an anomaly that never happened from a single camera frame, or you
can *re-shoot* footage you already have under weather and lighting you don't.
This walkthrough does one of each.

```{nvlearning-meta}
- **Level:** Intermediate
- **Tracks:** Two independent tracks, completable in either order
- **Generation time:** About 3.7 minutes per EVG run
- **Working directory:** `skill_walkthrough/`
```

## Learning Objectives

```{nvlearning-objectives}
- **Install** and read the two PAIDF local agent skills.
- **Evaluate** a seed image and a cookbook sampling configuration before running a generation job.
- **Generate** a traffic-anomaly video from a single seed frame with Event Video Generation.
- **Augment** an existing clip into new weather and time-of-day conditions with Video Data Augmentation.
- **Verify** generated output against the attribute verification table and the cookbook's documented failure modes.
```

## The Two Tracks

| Track | Pipeline | What it produces | Skill used |
|---|---|---|---|
| **A** | Event Video Generation (EVG) | A new traffic-anomaly video (a vehicle accident or a stalled vehicle) from a single camera frame | `physical-ai-event-video-generation-local` |
| **B** | Video Data Augmentation (VDA) | An existing traffic clip re-shot in different weather and time of day, plus auto-labels | `physical-ai-video-data-augmentation-local` |

Do them in either order. Each track follows the same shape: deploy the
pipeline's model endpoints, generate one new video, then review the result
against pre-generated reference videos.

:::{warning}
**GPU reality check.** A local EVG deploy needs 2× H100-class GPUs minimum —
one for Cosmos3-Nano, one for Nemotron. VDA needs one free GPU. If you don't
have that available, ask the agent for hosted endpoints instead.
:::

## Step 1: Install the Skills

Open your agent in the `skill_walkthrough/` directory. Its `skills/`
subdirectory already contains both skills this lab uses — nothing to clone or
register. Paste into the agent:

```text
Read skills/physical-ai-event-video-generation-local/SKILL.md and
skills/physical-ai-video-data-augmentation-local/SKILL.md. Confirm you've
read both.
```

## Step 2: Review the Reference Assets

Do this before you generate anything. Both pipelines are cheap to run and
expensive to run *wrong*, and in both cases the judgment call happens before
the job starts.

### The EVG Seed Frame

EVG needs one fixed-view traffic-cam frame per run — a highway, an
intersection, a parking lot. Ask the agent to open one and describe what makes
it usable: **wide fixed angle**, **visible lanes**, and **no camera hardware or
timestamps in frame**.

EVG's per-run choices are just `anomaly_type` (`vehicle_stopped` or
`accident`) and one fixed `env_type` — both spelled out in the prompts below,
so there's no config worth reading first. The seed frame itself is the only
real judgment call.

### The VDA Cookbook

VDA uses the `city_traffic` cookbook — an elevated camera over a multi-lane
intersection. Ask the agent to pull a frame from the demo clip (or your own
footage) the same way you just did for EVG's seed image.

Then look at the cookbook's actual sampling config rather than its full README.
`assets/cookbooks/city_traffic/workflow_config.yaml` is short enough to read
directly:

```yaml
augmentation:
  n_augmentations: 1
  variables:
    weather:
      clear: 0.35
      overcast: 0.35
      rain: 0.30
    time_of_day:
      morning: 0.25
      midday: 0.25
      evening: 0.25
      night: 0.25
```

That's what actually drives a run: `weather` and `time_of_day` are sampled from
these weights unless you force one.

Ask the agent to also pull the cookbook's documented failure modes from its
README — a couple of lines, not the whole file. For `city_traffic` these are
**overpass shadow confusing lighting assessment** and **ambiguous signal state
producing noisy red-light-violation calls**. You'll cross-check the auto-labels
against these later, so it's worth knowing them now rather than discovering
them after the fact.

If you have your own traffic footage, give the agent the local path instead of
the demo clip.

## Track A: Event Video Generation

### Step 3: Deploy the Model Endpoints

```text
Run preflight for physical-ai-event-video-generation-local, then deploy
the model endpoints it needs.
```

This deploys Nemotron 3 Nano Omni (VLM) and Cosmos 3 Nano, or prints hosted
export lines if you ask for `--hosted`.

### Step 4: Generate a Traffic-Anomaly Video

```text
Set up the EVG run directory with my seed images at /path/to/seed_images,
then generate a clip of a stalled vehicle on an elevated highway using
highway.png as the seed image, seed 42.
```

Or ask for the other supported type:

```text
Generate a clip of a vehicle accident on an elevated highway using
highway.png as the seed image.
```

**Expected:** roughly 3.7 minutes end to end, dominated by about 193 seconds of
Cosmos I2V inference. Each run produces four files:

```text
data/out/
├── highway_vehicle_stopped_000.mp4
├── highway_vehicle_stopped_000_prompt.txt
├── highway_vehicle_stopped_000_metadata.json
└── highway_vehicle_stopped_000_evaluation.json
```

### Step 5: Review the Result

```text
Evaluate the result — show me the prompt that was sent to Cosmos I2V and
the verification pass/fail table.
```

Read back the exact prompt Cosmos I2V received and the
`attribute_verification` evaluation pass/fail table — a VLM-answered check on
whether the requested anomaly, environment, and visual clarity actually show up.

:::{important}
A failed check means regenerate with a different seed. It does not mean ship
as-is. Synthetic data that doesn't contain the event you asked for is worse
than no data, because it teaches the model the wrong thing.
:::

Optionally annotate the clip:

```text
Run annotation on the video I just generated.
```

## Track B: Video Data Augmentation

### Step 6: Deploy the Model Endpoint

```text
Run preflight for physical-ai-video-data-augmentation-local, then deploy
the model endpoint it needs.
```

This deploys Nemotron 3 Nano Omni as a single shared VLM and LLM endpoint, or
reuses the instance Track A already started if you ran that first.

### Step 7: Re-Shoot the Clip in Different Conditions

```text
Set up a run for the city_traffic cookbook using the VDA demo video (or my
video at <path>), then generate a clear, nighttime version of it.
```

Then auto-label the result:

```text
Now auto-label the augmented clip.
```

Two things worth knowing before this runs:

- **There is no pre-populated model cache.** Cosmos Transfer weights download
  on first run into your local Hugging Face cache. The first run is slower than
  later ones, and you'll need disk space plus the gated Cosmos weights' license
  accepted.
- **Augmentation and auto-labeling each need a GPU.** No local GPU? Ask for
  hosted endpoints instead.

### Step 8: Review the Result

```text
Show me what got generated and render a side-by-side comparison against
the original.
```

This prints the sampled `weather` and `time_of_day` from `metadata.json`,
summarizes the auto-labeling output — detection and tracking, per-track
attributes, scene captions, and the anomaly-category vote across 10 event types
in 4 categories (`collision`, `near_miss`, `anomaly`, `normal_traffic`) — and
renders a side-by-side comparison video.

Cross-reference the labels against the `city_traffic` cookbook's documented
failure modes from Step 2. Treat auto-labels as a starting point for review,
not as ground truth.

```{nvlearning-checkpoint} Checkpoint 2
- Both PAIDF skills read and confirmed by the agent.
- Seed frame and cookbook sampling config reviewed before generation.
- Track A: one anomaly video generated from a seed frame, with its prompt and attribute verification table reviewed.
- Track B: one augmented clip generated and auto-labeled, with labels cross-checked against the cookbook's documented failure modes.
```

## Troubleshooting

| It reports | What is actually true |
|---|---|
| Preflight fails on available GPUs | Local EVG needs two H100-class GPUs, VDA needs one. Request hosted endpoints instead of deploying locally. |
| The first VDA run is far slower than expected | There is no pre-populated model cache. Cosmos Transfer weights download into the local Hugging Face cache on first run only. |
| Cosmos Transfer weights fail to download | The Cosmos weights are gated. Accept the license and confirm sufficient local disk space. |
| `attribute_verification` shows a failed check | The requested anomaly, environment, or visual clarity did not show up in the output. Regenerate with a different seed. |
| Auto-labels report odd lighting on an overpass | A documented `city_traffic` failure mode: overpass shadow confuses lighting assessment. |
| Auto-labels report noisy red-light violations | A documented `city_traffic` failure mode: ambiguous signal state. |

## Beyond This Lab: OSMO and Kubernetes

Neither track uses OSMO by default.
`physical-ai-video-data-augmentation-local` is a local-Docker counterpart of
the upstream `physical-ai-video-data-augmentation` skill, which submits work to
an OSMO cluster instead. Same worker scripts, cookbooks, and container images —
only the orchestration layer differs (`docker run` versus
`osmo workflow submit`). If you get access to an OSMO cluster, switch to the
upstream skill; every prompt in this walkthrough works unchanged against it.

EVG has two other execution paths this walkthrough doesn't cover: an
Airflow-DAG-on-Kubernetes path
(`paidf-orchestration/skills/physical-ai-event-video-generation/`, JSON
payload, `anomaly_dataset/` output layout) for Kubernetes and Airflow
environments, and the gated NGC skill bundle.

## What's Next

You now have the two data-generation levers that feed a training set. In
[Part 2.3: Fine-Tune for Alert Verification With TAO](part-2-3-fine-tune-for-alert-verification-with-tao)
you measure exactly what the base model gets wrong, then fine-tune it on the
traffic dataset and prove the improvement.
