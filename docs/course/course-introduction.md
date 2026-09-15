# Course Introduction

Traffic cameras already watch every major intersection in a modern city. The
hard part isn't seeing the road — it's deciding which of the thousands of
things that happen on it are worth waking someone up for. A pipeline that
flags every stopped car floods the operator with noise; one tuned to stay
quiet misses the collision that mattered.

In this course you build a vision AI agent that resolves that tension. You
start from a stock deployment, measure exactly where it fails, generate the
data it's missing, fine-tune the model that verifies its alerts, and redeploy
the improved system — the full loop, end to end, driven by coding agents and
Agent Skills.

```{nvlearning-meta}
- **Level:** All levels — no prior VLM or Cosmos experience needed
- **Audience:** Developers building vision AI agents and video analytics applications
- **Structure:** Part 1 (three notebooks) and Part 2 (four agent-driven walkthroughs)
- **Primary model:** Cosmos 3 Reasoner
- **Anomaly classes:** `collision` and `stalled` (a vehicle stopped in a travel lane)
```

## Learning Objectives

```{nvlearning-objectives}
- **Apply** the Cosmos 3 Reasoner and Generator towers to the jobs each one fits.
- **Generate** synthetic data for rare and dangerous events using NVIDIA Physical AI Data Factory and NVIDIA Cosmos.
- **Apply** LoRA fine-tuning to train efficient, deployable anomaly detection models.
- **Deploy** models using the NVIDIA Blueprint for Video Search and Summarization (VSS) with Alert Verification to reduce false positives.
- **Orchestrate** synthetic data generation, training, and deployment stages using Agent Skills.
```

## What You'll Build

You'll build a **Video Analytics AI Agent** on the NVIDIA Blueprint for Video
Search and Summarization (VSS). It takes multiple traffic camera streams as
input and emits high-confidence alerts when an anomaly occurs.

You don't build it from scratch. VSS ships the pieces as prebuilt,
GPU-accelerated microservices, and your job is to compose them, point them at
your data, and improve the one that limits your accuracy. The agent uses several
VSS components:

| VSS component | What it does for your agent |
|---|---|
| **RT-CV** | A prebuilt vision microservice for GPU-optimized detection and tracking of vehicles |
| **RT-VLM** | Serves the vision language model — **Cosmos 3 Reasoner** — that answers questions about a clip |
| **Behavior Analytics** | Turns raw tracks into candidate alerts: stopped vehicle, collision |
| **Alert Bridge** | Sends each candidate clip to the VLM and records the confirm-or-reject verdict |
| **VIOS** | Video ingest and storage — registers streams and keeps the clips retrievable |
| **Alert UI** | Where you read the alerts, filter by sensor, and watch the clip behind a verdict |

Those components form a three-stage pipeline:

1. **RT-CV** detects and tracks every vehicle in the stream.
2. **Behavior Analytics** turns those tracks into candidate alerts.
3. **Alert Bridge** sends each candidate to **Cosmos 3 Reasoner**, which
   confirms or rejects it.

That third stage is the one that controls your false positive rate, and it's
the one you'll improve. The computer vision stages find *everything* that
might be an event; the VLM decides what's real. Fine-tune the verifier and
you change the system's accuracy without touching detection or tracking.

## The Model at the Center

NVIDIA Cosmos is an open platform of world models, datasets, and tools for
building Physical AI. Cosmos 3 exposes two towers, and this course uses both:

| Tower | What it does | Where you use it |
|---|---|---|
| **Reasoner** | Takes text and vision, returns text | The alert verdict, and the model you fine-tune |
| **Generator** | Takes text, vision, sound, and action; returns vision, sound, and action | Synthetic training video for the gaps you find |

The specific model is **Cosmos 3 Reasoner** at the Nano size — small enough that
a full fine-tune-and-redeploy loop fits inside a lab session.

## Why the Course Is in Two Parts

Part 1 works at the **model level**. You deploy Cosmos 3 Reasoner, send it
images and video, learn what it can and can't do, shape its output, measure its
speed, and fine-tune it. Nothing sits between you and the model. By the end you
understand its core capabilities firsthand — and that understanding is what
makes everything in Part 2 legible instead of magical.

That is a legitimate way to use Cosmos: stand up the NIM, send it requests, get
answers back. It's also where the interesting questions start, because the next
ones aren't about the model at all. Part 2 answers them with **production
blueprints** — and with the Agent Skills that drive those blueprints for you.

### From One Request to a Running System

A camera doesn't hand you a clip. It emits an unbounded live stream. So:

- How do you connect the model to a **real-time streaming camera** and keep up
  with it?
- Where does the footage live afterward, and how do you search a **video
  database** of everything you've already processed?
- When the model returns a verdict, how does it reach the rest of your system —
  on a **scalable message bus** rather than a function call that blocks?

That is what **VSS** provides. It is the production blueprint that wraps the
model in ingest, storage, analytics, and messaging, so a single inference call
becomes a system that runs continuously across many cameras. The **VSS agent
skills** are how you drive that blueprint: you describe the deployment you want
and the agent knows which services, profiles, and rules to stand up.

### From One Generated Clip to a Training Set

The same gap opens on the data side. You can drive the Generator tower yourself
and get a video out of it. But a video is not training data:

- Who produces the **labels** that make a clip trainable?
- How do you check the **quality** of what came out — that the clip actually
  contains the event you asked for?
- How do you write **prompts** for thousands of variations and build a dataset
  at that **scale**, rather than one clip at a time?

That is what **PAIDF** — NVIDIA Physical AI Data Factory — provides. It is the
blueprint for generating data at scale: seeded generation, augmentation,
auto-labeling, and quality verification as repeatable stages instead of manual
steps.

### From a Notebook Fine-Tune to a Deployable Checkpoint

Fine-tuning in a notebook gets you weights. Serving them is a different job:
the run has to be reproducible, the result has to be measured against a held-out
set, and the adapter has to become an artifact your inference stack can load.

That is what **NVIDIA TAO Toolkit** provides. It fine-tunes Cosmos 3 Reasoner
with LoRA, evaluates the result, and merges the adapter into a plain checkpoint
that VSS can serve — the same recipe you ran by hand, as a job you can rerun.

## Course Structure

The two parts answer two different questions:

| | Question it answers | How you work |
|---|---|---|
| **Part 1 — Cosmos Basics** | What can this model do? | Notebooks, cell by cell, directly against the model |
| **Part 2 — Application** | How do I run this in production, at scale? | Coding agents driving the VSS, PAIDF, and TAO blueprints through Agent Skills |

We teach the model first on purpose. When the agent deploys 31 services in Part
2.1, you already know what the VLM at the center of them is doing — so you're
evaluating a system, not trusting a black box.

| Part | What you do |
|---|---|
| [Prerequisites](prerequisites) | Generate an NGC Personal API Key before the course |
| [Part 1](part-1-cosmos-basics) | Complete notebooks 1.1, 1.2, and 1.3 |
| [Part 2.1](part-2-1-deploy-zero-shot-vss) | Deploy zero-shot VSS for alert verification |
| [Part 2.2](part-2-2-expand-your-dataset-with-paidf) | Expand the dataset with PAIDF |
| [Part 2.3](part-2-3-fine-tune-for-alert-verification-with-tao) | *(Optional)* Fine-tune for alert verification with TAO — [video tutorial](https://youtu.be/9AQkVbx3fKA) |
| [Part 2.4](part-2-4-deploy-fine-tuned-vss) | Deploy fine-tuned VSS and compare |

## The Workflow You'll Follow

Every part of this course is one stage of a single improvement loop. It's the
loop you'd run against any vision AI system, not just this one:

1. **Define the problem.** Send high-confidence alerts when traffic anomalies
   occur.
2. **Create an evaluation dataset.** You can't improve what you don't measure.
3. **Run a zero-shot evaluation** on the models and the assembled system.
4. **Identify gaps** in model accuracy.
5. **Collect, augment, generate, and label** a training dataset targeting
   those gaps.
6. **Fine-tune** the VLM.
7. **Redeploy** VSS with the fine-tuned model and measure again.

## Two Evaluations, Measured Separately

This distinction matters more than any other idea in the course, and mixing
the two numbers up is the most common way to misread your own results.

| Evaluation | What it measures | Where you run it | Metric |
|---|---|---|---|
| **Model level** | Cosmos 3 Reasoner on 128 held-out clips | Part 2.3 | Exact-match `LABEL=` answers, macro-F1 |
| **System level** | The full pipeline on four evaluation videos | Parts 2.1 and 2.4 | Behavior Analytics candidates, Cosmos verdicts |

The model-level number tells you whether the weights improved. The
system-level number tells you whether the *product* improved. They move
together, but they are never equal, because Alert Bridge asks its own
question with its own frame sampling. Always say which one you mean.

## Requirements

```{nvlearning-requirements}
- Comfort reading Python and running commands in a terminal. You use Docker, but every command you need is given to you.
- No prior VLM, Cosmos, or fine-tuning experience needed. Part 1 builds it from the ground up.
- The course environment provides two H100-class GPUs, a remote Docker daemon, and all container images and model weights pre-staged.
- An **NGC Personal API Key**, generated before the course. Part 2.1 needs it for the tuned profile build; see [Prerequisites](prerequisites).
```

:::{note}
The Part 2 environment runs Docker-in-Docker: the Docker daemon is remote and
reached through `DOCKER_HOST`, not local to your shell. Several walkthrough
steps depend on that distinction. Each walkthrough gives the agent an explicit
context block covering it — paste those blocks verbatim.
:::

```{nvlearning-checkpoint} Before You Continue
You can state the three stages of the VSS alert verification pipeline, name
which stage you will fine-tune, and explain why the model-level and
system-level evaluations produce different numbers.
```

## What's Next

Check the [Prerequisites](prerequisites) first — you need an NGC Personal API
Key before Part 2.1, and it takes about five minutes to generate. Then start
with [Part 1: Cosmos Basics](part-1-cosmos-basics) to build the foundation, and
move into the agent-driven walkthroughs in Part 2.
