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
| `lab-1-vlm/` | Notebook 1.1 — Vision Language Models |
| `lab-2-sdg/` | Notebook 1.2 — Augmenting and Generating Datasets |
| `lab-3-vlm-pt/` | Notebook 1.3 — Fine-Tuning Vision Language Models |

## Course Structure

**Part 1 — Cosmos Basics** runs from the three notebook folders above.

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

## Publish the Docs to the DLI Course

The built site is delivered to learners inside JupyterLab in the DLI course
environment (`gitlab.com/nvidia/dli/content/x-fx-138-v1`). It is **not** served
from this repository. Publishing is a manual step: build here, upload to S3, and
the course's S3 data loader copies it into the lab at launch.

### How it reaches the learner

```
this repo  ->  docs/_build/html        (sphinx-build)
           ->  s3://dli-lms/data/x-fx-138-v1-docs/course-docs/   (aws s3 cp)
           ->  /dli/task/data/course-docs/                       (S3 data loader, at launch)
           ->  /lab/course-docs/                                 (jupyter-server-proxy)
           ->  the "Course Docs" tile in the JupyterLab Launcher
```

The DLI side of that chain lives in `x-fx-138-v1`: `task1/lab/Dockerfile`
installs `jupyter-server-proxy`, `task1/lab/configs/jupyter_configs/jupyter_server_config.py`
defines the proxied server and its `DOCS_ROOT`, and `task1/data_sources` lists
the S3 prefix.

### 1. Build the public edition

Publish the **public** edition only. The internal edition contains facilitator
material (shot lists, instructor cues, the Part 2.3 time budget) and must not be
uploaded.

```bash
uv sync
rm -rf docs/_build
uv run sphinx-build -b html -W --keep-going docs docs/_build/html
```

`-W --keep-going` is not optional here: a broken cross-reference that only warns
locally becomes a dead link for every learner.

### 2. Trim what is never served

About 4 MB of the build is never requested by a browser. Removing it is safe:

```bash
rm -rf docs/_build/html/.doctrees docs/_build/html/.buildinfo docs/_build/html/_sources
find docs/_build/html -name '*.map' -delete
```

- `.doctrees/` — Sphinx build cache
- `_sources/` — nothing in the generated HTML links to it
- `*.map` — sourcemaps, fetched only when devtools is open

### 3. Upload

Run this from a machine with AWS credentials for the `dli-lms` bucket.

```bash
aws s3 cp --recursive --acl public-read \
  docs/_build/html/ \
  s3://dli-lms/data/x-fx-138-v1-docs/course-docs/
```

The trailing slashes matter. `--acl public-read` matches how every other course
prefix is uploaded.

To move the build to another machine first:

```bash
tar -czf course-docs.tar.gz -C docs/_build html && mv course-docs.tar.gz ~/
# then, on the machine with credentials:
tar -xzf course-docs.tar.gz    # creates ./html/
aws s3 cp --recursive --acl public-read html/ \
  s3://dli-lms/data/x-fx-138-v1-docs/course-docs/
```

Upload the **extracted files**, not the tarball. The DLI data loader copies keys
verbatim and does not unpack archives.

### 4. Verify

```bash
aws s3 ls --recursive s3://dli-lms/data/x-fx-138-v1-docs/ | wc -l
aws s3 ls s3://dli-lms/data/x-fx-138-v1-docs/course-docs/ | head
```

Expect roughly 69 objects and `index.html` at the top of `course-docs/`.

### 5. Enable delivery (first publish only)

In `x-fx-138-v1`, uncomment the last line of `task1/data_sources`:

```
--recursive s3://dli-lms/data/x-fx-138-v1-docs/
```

> **Do this only after step 4 confirms objects exist.** That file's own warning
> applies: listing an empty S3 prefix makes the loader raise `KeyError` and stop
> processing, which breaks every course launch.

Once the line is live, later publishes are just steps 1-4 — no DLI-side change,
and no image rebuild.

### Notes

- **Use a sibling prefix, not the main course prefix.** `x-fx-138-v1-docs/` is
  deliberately separate from `x-fx-138-v1/`. The loader lists at most 1000 keys
  per `--recursive` line and the main course prefix already overflows that, so
  adding ~69 doc objects there would silently displace other assets. See the
  1000-key notes in `task1/data_sources`.
- **Rebuilds are not automatic.** Learners see whatever was last uploaded.
  Re-run steps 1-4 after merging doc changes you want them to have.
- **The tile reads the live directory**, so a re-upload shows up on the next
  course launch without touching the lab image.

## Authoring Notes

- Course pages live in `docs/course/` and are listed in the `docs/index.md`
  toctree in curriculum order.
- Gate facilitator content with `:::{only} internal`. Do **not** put a section
  heading inside an `{only}` block — headings leak into non-matching builds.
  Use a bold label instead.
