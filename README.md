# Singapore AI Day DLI

**Build High-Accuracy Vision AI Agents for Anomaly Detection Using Synthetic
Data and Fine-Tuning**

An end-to-end course on building a vision AI agent for smart-city traffic
anomaly detection with NVIDIA Cosmos, the NVIDIA Blueprint for Video Search and
Summarization (VSS), NVIDIA Physical AI Data Factory, and NVIDIA TAO Toolkit.

## Repository Layout

| Path | Contents |
|---|---|
| `docs/` | The course walkthrough site (Sphinx + `sphinx-nvlearning`) |
| `notebook-1-1-vlm/` | Notebook 1.1 — Vision Language Models |
| `notebook-1-2-sdg/` | Notebook 1.2 — Augmenting and Generating Datasets |
| `notebook-1-3-vlm-pt/` | Notebook 1.3 — Fine-Tuning Vision Language Models |

## Course Structure

**Part 1 — Theory and Concepts** runs from the three notebook folders above.

**Part 2 — Application** is four agent-driven walkthroughs, documented in
`docs/course/`:

| Page | Topic |
|---|---|
| Part 2.1 | Deploy zero-shot VSS for alert verification |
| Part 2.2 | Augment the dataset with PAIDF (EVG and VDA) |
| Part 2.3 | Fine-tune for alert verification with TAO |
| Part 2.4 | Deploy fine-tuned VSS and compare results |

## Build the Documentation

The site uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run sphinx-build -b html docs docs/_build/html
uv run python -m http.server 8000 -d docs/_build/html/
```

Then open `http://localhost:8000`.

One-command alternative that builds, serves, and opens a browser:

```bash
uv run python scripts/preview_learning_site.py
```

## Editions

The site builds in multiple editions from one source:

- `nvlearning_mode`: `course`, `lab`, or `hub`
- `nvlearning_audience`: `public` or `internal`

Facilitator material — shot lists, instructor cues, and the Part 2.3 time
budget — is gated to the `internal` audience and does **not** appear in the
default build. To produce the internal edition:

```bash
uv run sphinx-build -b html -D nvlearning_audience=internal docs docs/_build/internal
```

Add `-W --keep-going` to fail the build on warnings for release-quality output.

## Authoring Notes

- Course pages live in `docs/course/` and are listed in the `docs/index.md`
  toctree in curriculum order.
- Gate facilitator content with `:::{only} internal`. Do **not** put a section
  heading inside an `{only}` block — headings leak into non-matching builds.
  Use a bold label instead.
