# Build High-Accuracy Vision AI Agents for Anomaly Detection

Build a **Video Analytics AI Agent** for smart-city traffic anomaly detection
with NVIDIA Cosmos, the NVIDIA Blueprint for Video Search and Summarization
(VSS), NVIDIA Physical AI Data Factory (PAIDF), and NVIDIA TAO Toolkit.

You start at the model level with Cosmos, then move to the production
blueprints — driven by coding agents and Agent Skills. Along the way you take a
stock deployment, measure exactly where it fails, generate the data it's
missing, fine-tune the model that verifies its alerts, and redeploy the
improved system.

```{nvlearning-meta}
- **Level:** All levels — no prior VLM or Cosmos experience needed
- **Audience:** Developers building vision AI agents and video analytics applications
- **Structure:** Part 1 (three notebooks) and Part 2 (four agent-driven walkthroughs)
```

## Start Here

Begin at the [Course Introduction](course/course-introduction.md) for what
you'll build, why the course is split in two, and the improvement loop that
structures it. Then work through Part 1 and the four Part 2 walkthroughs in
order.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Course Introduction
:link: course/course-introduction
:link-type: doc

The Video Analytics AI Agent you'll build, the VSS components it uses, and
why the course moves from model level to blueprint level.
:::

:::{grid-item-card} Part 1: Cosmos Basics
:link: course/part-1-cosmos-basics
:link-type: doc

Cosmos at the model level. Three notebooks covering the Reasoner and
Generator towers, generative augmentation, and VLM fine-tuning.
:::

:::{grid-item-card} Part 2: Application
:link: course/part-2-1-deploy-zero-shot-vss
:link-type: doc

The blueprints at production scale. Four walkthroughs driven by coding agents
and Agent Skills — deploy, expand, fine-tune, redeploy.
:::

:::{grid-item-card} Conclusion
:link: course/conclusion
:link-type: doc

Results recap, VSS profiles, and where to take the workflow next.
:::

::::

## Course Notebooks

Part 1 runs from the notebook folders in this repository:

| Notebook | Folder |
|---|---|
| 1.1 Vision Language Models | `notebook-1-1-vlm/lab_1.ipynb` |
| 1.2 Augmenting and Generating Datasets | `notebook-1-2-sdg/sdg_part1_notebook.ipynb` |
| 1.3 Fine-Tuning Vision Language Models | `notebook-1-3-vlm-pt/lab_3.ipynb` |

```{toctree}
:maxdepth: 2
:hidden:

Course Introduction <course/course-introduction>
Part 1: Cosmos Basics <course/part-1-cosmos-basics>
Part 2.1: Deploy Zero-Shot VSS for Alert Verification <course/part-2-1-deploy-zero-shot-vss>
Part 2.2: Expanding Your Dataset With PAIDF <course/part-2-2-expand-your-dataset-with-paidf>
Part 2.3 (Optional): Fine-Tune for Alert Verification With TAO <course/part-2-3-fine-tune-for-alert-verification-with-tao>
Part 2.4: Deploy Fine-Tuned VSS for Alert Verification <course/part-2-4-deploy-fine-tuned-vss>
Conclusion <course/conclusion>
```
