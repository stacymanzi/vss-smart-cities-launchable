# Part 1: Theory and Concepts

Part 1 is hands-on notebook work. You deploy a vision language model, prompt
it well and badly, generate synthetic video, and fine-tune a checkpoint — each
one a piece of the system you assemble in Part 2.

The teaching content lives in the notebooks themselves. This page tells you
what to run, in what order, and what you should be able to do when you finish.

```{nvlearning-meta}
- **Level:** Intermediate
- **Format:** Three Jupyter notebooks, run cell by cell
- **Location:** `/dli/task/`
```

## Learning Objectives

```{nvlearning-objectives}
- **Deploy** Cosmos Reason 3 Nano and run inference on both images and video.
- **Distinguish** effective from ineffective VLM prompts, and structure model output as formatted reports, JSON, and true/false or multiple-choice answers.
- **Benchmark** a deployed VLM for latency and throughput across sampled frames, input and output sequence length, and concurrency.
- **Generate** synthetic video with Cosmos using image-to-image, image-to-video, and video-to-video workflows.
- **Organize** a dataset for VLM fine-tuning and apply LoRA supervised fine-tuning to Cosmos Reason 3 Nano.
- **Evaluate** a fine-tuned VLM using appropriate task types, annotations, and metrics.
```

## Complete the Three Notebooks

Run them in order. Each one assumes the previous is finished.

### Notebook 1.1 — Vision Language Models

Deploy Cosmos Reason 3 Nano and learn how it behaves. You'll run inference on
images and videos, see how frames become vision tokens, compare good and bad
prompts, and force structured outputs — formatted reports, JSON, true/false,
and multiple choice. You'll also compare reasoning against non-reasoning modes
and benchmark performance for latency and throughput. Tear down the deployment
at the end.

### Notebook 1.2 — Augmenting and Generating Datasets With Generative Models

Deploy Cosmos for video generation and produce your own training data. You'll
choose a seed image — real or synthetic — and work through the configurations,
prompts, and best practices for image-to-image, image-to-video, and
video-to-video generation. You'll augment a seed image with Qwen Image Edit,
generate video from it with Cosmos, and augment existing video with
video-to-video. You'll also use an LLM inside the notebook to help write
generation prompts.

### Notebook 1.3 — Fine-Tuning Vision Language Models

Organize a dataset for VLM fine-tuning and post-train Cosmos Reason 3 Nano.
You'll cover the task types worth fine-tuning on, compare fine-tuning methods
— SFT, LoRA, and RL — and work through VLM evaluation: task types, annotation
types, and metric selection. You finish by fine-tuning Cosmos Nano itself.

:::{important}
Part 2.3 runs the same LoRA recipe you run here — same rank, same target
modules, same hyperparameters — but through Agent Skills instead of notebook
cells. Finishing 1.3 is what makes Part 2.3 a comparison rather than a first
encounter.

Part 2.3 does **not** depend on it, though. That walkthrough is self-contained
and ships its own configuration; nothing in Part 2 reads this notebook's output
or requires it to have been run. If you fall behind or restart your instance,
you can start any Part 2 walkthrough directly.
:::

:::{only} internal
Presenter assignments for Part 1: notebook 1.1 Vision Language Models, notebook
1.2 Augmenting and Generating Datasets, notebook 1.3 Fine-tuning Vision
Language Models. The opening Physical AI, Cosmos, VSS, PAIDF, and TAO overview
is delivered as a presentation before notebook 1.1.
:::

```{nvlearning-checkpoint} Part 1 Complete
- Notebook 1.1 run end to end, with the Cosmos Reason deployment torn down.
- Notebook 1.2 run end to end, with at least one generated video reviewed.
- Notebook 1.3 run end to end, with a fine-tuned checkpoint and its evaluation metrics.
- You can explain what LoRA changes and what it leaves frozen.
```

## What's Next

Part 2 rebuilds this same workflow with coding agents. Start with
[Part 2.1: Deploy Zero-Shot VSS for Alert Verification](part-2-1-deploy-zero-shot-vss),
where you stand up the traffic pipeline and find out where a stock deployment
falls short.
