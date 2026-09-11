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
- **Level:** Intermediate
- **Audience:** Developers building vision AI agents and video analytics applications
- **Structure:** Part 1 (three notebooks) and Part 2 (four agent-driven walkthroughs)
- **Anomaly classes:** `collision` and `stalled` (a vehicle stopped in a travel lane)
```

## Learning Objectives

```{nvlearning-objectives}
- **Generate** synthetic data for rare and dangerous events using NVIDIA Physical AI Data Factory and NVIDIA Cosmos.
- **Apply** LoRA fine-tuning to train efficient, deployable anomaly detection models.
- **Deploy** models using the NVIDIA Blueprint for Video Search and Summarization (VSS) with Alert Verification to reduce false positives.
- **Orchestrate** synthetic data generation, training, and deployment stages using Agent Skills.
```

## What You'll Build

The system takes multiple traffic camera streams as input and emits
high-confidence alerts when an anomaly occurs. It is built with VSS as a
three-stage pipeline:

1. **RT-CV** detects and tracks vehicles using the DeepStream SDK.
2. **Behavior Analytics** turns those tracks into candidate alerts — stopped
   vehicle, collision.
3. **Alert Bridge** sends each candidate clip to Cosmos Reason 3 Nano, which
   confirms or rejects it.

That third stage is the one that controls your false positive rate, and it's
the one you'll improve. The computer vision stages find *everything* that
might be an event; the VLM decides what's real. Fine-tune the verifier and
you change the system's accuracy without touching detection or tracking.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} NVIDIA Cosmos
An open platform of world models for Physical AI. The **Reasoner** surface
takes text and vision and produces text — the verification verdict. The
**Generator** surface produces video — your synthetic training data.
:::

:::{grid-item-card} VSS Blueprint
Reference architectures for building vision agents, composed of
GPU-accelerated microservices orchestrated over the Model Context Protocol
(MCP) and deployable on premises.
:::

:::{grid-item-card} Physical AI Data Factory
Agent-driven workflows for curating, enriching, labeling, augmenting, and
generating physical AI data. You'll use Event Video Generation and Video
Data Augmentation.
:::

:::{grid-item-card} NVIDIA TAO Toolkit
Fine-tunes Cosmos Reason with LoRA, evaluates the result, and merges the
adapter into a plain checkpoint that VSS can serve.
:::

::::

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
| **Model level** | Cosmos Reason on 128 held-out clips | Part 2.3 | Exact-match `LABEL=` answers, macro-F1 |
| **System level** | The full pipeline on four evaluation videos | Parts 2.1 and 2.4 | Behavior Analytics candidates, Cosmos verdicts |

The model-level number tells you whether the weights improved. The
system-level number tells you whether the *product* improved. They move
together, but they are never equal, because Alert Bridge asks its own
question with its own frame sampling. Always say which one you mean.

## Course Structure

**Part 1 — Theory and Concepts** covers vision language models, generative
augmentation, and VLM fine-tuning across three Jupyter notebooks. You run the
code cell by cell and build the mental model.

**Part 2 — Application** rebuilds that same workflow as four walkthroughs
driven by a coding agent and Agent Skills. Instead of running cells, you tell
an agent what you want and it reads the relevant skill to learn how.

| Part | What you do |
|---|---|
| [Part 1](part-1-theory-and-concepts) | Complete notebooks 1.1, 1.2, and 1.3 |
| [Part 2.1](part-2-1-deploy-zero-shot-vss) | Deploy zero-shot VSS for alert verification |
| [Part 2.2](part-2-2-expand-your-dataset-with-paidf) | Expand the dataset with PAIDF |
| [Part 2.3](part-2-3-fine-tune-for-alert-verification-with-tao) | Fine-tune for alert verification with TAO |
| [Part 2.4](part-2-4-deploy-fine-tuned-vss) | Deploy fine-tuned VSS and compare |

## Requirements

```{nvlearning-requirements}
- Working knowledge of Python, Docker, and the command line.
- Familiarity with deep learning concepts; prior VLM experience is helpful but not required.
- The course environment provides two H100-class GPUs, a remote Docker daemon, and all container images and model weights pre-staged.
- No NGC key or Hugging Face token is needed for Part 2.3. Part 2.1 uses an NGC personal key for the tuned profile build.
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

Start with [Part 1: Theory and Concepts](part-1-theory-and-concepts) to build
the foundation, then move into the agent-driven walkthroughs in Part 2.
