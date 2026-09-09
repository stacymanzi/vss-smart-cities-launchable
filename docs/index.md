# Build High-Accuracy Vision AI Agents for Anomaly Detection

Build an end-to-end vision AI agent for smart-city traffic anomaly detection
with NVIDIA Cosmos, the NVIDIA Blueprint for Video Search and Summarization
(VSS), and Metropolis Skills.

You start from a stock deployment, measure exactly where it fails, generate the
data it's missing, fine-tune the model that verifies its alerts, and redeploy
the improved system.

```{nvlearning-meta}
- **Level:** Intermediate
- **Audience:** Developers building vision AI agents and video analytics applications
- **Structure:** Part 1 (three notebooks) and Part 2 (four agent-driven walkthroughs)
```

## Start Here

Begin at the [Course Introduction](course/course-introduction.md) for the
architecture, the workflow, and how the two evaluations differ. Then work
through Part 1 and the four Part 2 walkthroughs in order.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Course Introduction
:link: course/course-introduction
:link-type: doc

What you'll build, the three-stage alert verification pipeline, and the
improvement loop that structures the whole course.
:::

:::{grid-item-card} Part 1: Theory and Concepts
:link: course/part-1-theory-and-concepts
:link-type: doc

Three notebooks: vision language models, generative augmentation, and VLM
fine-tuning.
:::

:::{grid-item-card} Part 2: Application
:link: course/part-2-1-deploy-zero-shot-vss
:link-type: doc

Four walkthroughs driven by coding agents and Agent Skills — deploy, augment,
fine-tune, redeploy.
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

## Build This Site

```bash
uv sync
uv run sphinx-build -b html docs docs/_build/html
uv run python -m http.server 8000 -d docs/_build/html/
```

Then open `http://localhost:8000`.

One-command alternative that builds, serves, and opens a browser (press
`Ctrl+C` to stop):

```bash
uv run python scripts/preview_learning_site.py
```

## Build the Internal Edition

Facilitator material — shot lists, instructor cues, and the Part 2.3 time
budget — is gated to the internal audience and does not appear in the default
build. To produce it:

```bash
uv run sphinx-build -b html -D nvlearning_audience=internal docs docs/_build/internal
```

- `nvlearning_mode`: `course`, `lab`, or `hub`
- `nvlearning_audience`: `public` or `internal`

Add `-W --keep-going` to fail the build on warnings.

```{toctree}
:maxdepth: 2
:hidden:

Course Introduction <course/course-introduction>
Part 1: Theory and Concepts <course/part-1-theory-and-concepts>
Part 2.1: Deploy Zero-Shot VSS for Alert Verification <course/part-2-1-deploy-zero-shot-vss>
Part 2.2: Augment the Dataset With PAIDF <course/part-2-2-augment-the-dataset-with-paidf>
Part 2.3: Fine-Tune for Alert Verification With TAO <course/part-2-3-fine-tune-for-alert-verification-with-tao>
Part 2.4: Deploy Fine-Tuned VSS for Alert Verification <course/part-2-4-deploy-fine-tuned-vss>
Conclusion <course/conclusion>
```
