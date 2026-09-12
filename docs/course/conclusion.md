# Conclusion

You started with a stock VSS deployment that produced zero traffic candidates
and finished with a fine-tuned pipeline that recovers a collision the base
model missed, with no new false positives. Everything in between was one loop,
run once, with measurements at both ends.

```{nvlearning-meta}
- **Level:** All levels — no prior VLM or Cosmos experience needed
- **Covers:** Course recap, the model-to-blueprint arc, VSS profiles, and where to go next
```

## What You Built

A **Video Analytics AI Agent** for smart-city traffic anomaly detection, built
on the NVIDIA Blueprint for Video Search and Summarization (VSS) from prebuilt,
GPU-accelerated microservices:

1. **RT-CV**, a prebuilt vision microservice from VSS, performs GPU-optimized
   detection and tracking of vehicles.
2. **Behavior Analytics** converts tracks into candidate alerts for collisions
   and stalled vehicles.
3. **Alert Bridge** sends each candidate clip to **Cosmos 3 Reasoner**, served
   by RT-VLM, for a confirm-or-reject verdict.

Around them, **VIOS** handled ingest and storage and the **Alert UI** gave you
somewhere to read verdicts and watch the clip behind each one.

You changed exactly one component — the verifier — and measured what that was
worth.

## From the Model to the Blueprints

The course was built in two parts on purpose, and it's worth naming what each
one gave you now that you've done both.

**Part 1 worked at the model level.** You deployed Cosmos 3 Reasoner, prompted
it well and badly, forced its output into a fixed shape, measured its latency
and throughput, drove the Generator tower, and fine-tuned a checkpoint. Nothing
sat between you and the model.

**Part 2 worked at the blueprint level**, because the questions that come next
aren't model questions at all:

| The model-level version | The production version | Blueprint |
|---|---|---|
| Send a clip, read the answer | Run continuously against live camera streams, store the footage, publish verdicts to other systems | **VSS** |
| Generate one video | Produce labeled, quality-checked datasets at scale | **PAIDF** |
| Fine-tune in a notebook | Launch, evaluate, and merge a reproducible job that yields a servable checkpoint | **TAO** |

Neither level is sufficient alone. Without Part 1 you'd be deploying a black
box and trusting its verdicts. Without Part 2 you'd have a model that answers
one clip at a time and no way to put it in front of a camera.

## What Agent Skills Bought You

Every blueprint in Part 2 was driven by a coding agent reading Agent Skills, and
that's the reason the whole loop fit in one session rather than a sprint.

Deploying VSS by hand means reading deployment docs, choosing a profile,
resolving host and external addresses, composing 31 services, registering
streams, and authoring alert rules — then doing it again when a setting is
wrong. You described what you wanted instead, and the skill supplied the
expertise: which parameters matter, what healthy looks like, which warnings are
expected and which are real.

You stayed the decision-maker throughout. Every container asked for approval,
and the command it showed you was the lesson. What you skipped was having to
learn three toolchains before you could test a single idea — which is exactly
what makes running the improvement loop affordable enough to actually do.

## The Results

**Model level**, on 128 held-out clips:

| Metric | Base | Fine-tuned |
|---|---|---|
| Macro-F1 | 0.733 | 0.961 |
| Precision | 1.000 | 0.928 |
| Recall | 0.500 | 1.000 |
| `stalled` detected | 0/32 | 32/32 |

**System level**, on four evaluation videos covering 16 events:

| Metric | Stock | Fine-tuned |
|---|---|---|
| Behavior Analytics coverage | 16/16 | 16/16 |
| Cosmos confirmed | 14/16 | 15/16 |
| Confirmed false positives | 0 | 0 |

Two numbers, two meanings. The model-level jump is large because the base model
never predicted `stalled` at all — a class gap that fine-tuning closed
completely. The system-level jump is one event, because Alert Bridge asks its
own question with its own frame sampling. Reporting the 0.228 macro-F1 gain as
if it were a product improvement would be wrong; so would dismissing a
recovered collision as "only one event."

## What the Numbers Taught You

Three things worth carrying to the next project:

- **A gap has a shape.** The base model scored identically across original,
  fog, rain, and night. That uniformity is what identified it as a class
  problem rather than a weather problem — and told you that no amount of prompt
  engineering would fix it.
- **Fixing one gap can open another.** After fine-tuning, clean clips were
  perfect while fog, rain, and night trailed slightly. That residual is not a
  failure; it's the specification for the next data generation run.
- **Constrain the output to make evaluation cheap.** A single `LABEL=<class>`
  line is what makes exact-match scoring valid. Design the answer contract
  before you design the metric.

## VSS Profiles

A profile is how VSS packages a deployment for a use case: which models load,
which CV configuration runs, which analytics rules are enabled, and how many
services come up. You worked with three variations of the same blueprint:

| Profile | What it gave you |
|---|---|
| Stock alerts profile, verification mode | About 31 services, warehouse ladder/PPE rules, a stopped-vehicle rule needing 120 s of stillness — zero traffic candidates |
| Tuned AI City traffic profile | 19 services, traffic-appropriate CV and thresholds — candidates and verdicts on real events |
| Tuned profile with `--vlm-model-dir` | The same 19 services with a fine-tuned verifier swapped in via one build flag |

The progression is the lesson. A stock profile is a starting point, not a
product. Tuning the profile fixed *detection*; fine-tuning the model fixed
*verification*. They're separate problems with separate fixes, and diagnosing
which one you have is most of the work.

## Continue the Loop

The workflow doesn't end at one iteration. The weather residual you measured in
Part 2.3 is a concrete target: generate more fog, rain, and night clips with
the PAIDF tracks from Part 2.2, fold them into the training split, and run the
same evaluation again. Same loop, better data, measurable result.

And nothing about that loop is specific to traffic. Swap the anomaly classes,
the seed imagery, and the evaluation set, and the same three blueprints carry a
different Video Analytics AI Agent from baseline to measured improvement.

## Where to Go Next

- [NVIDIA Cosmos](https://github.com/nvidia/cosmos) — world foundation models,
  the Reasoner and Generator towers, and model sizes from Edge to Super.
- [NVIDIA Blueprint for Video Search and Summarization](https://docs.nvidia.com/vss/latest/) —
  full reference architectures, microservice details, and deployment options.
- [NVIDIA Physical AI Data Factory](https://github.com/NVIDIA/physical-ai-data-factory) —
  the EVG and VDA workflows, cookbooks, and auto-labeling modules.
- [NVIDIA TAO Toolkit VLM fine-tuning](https://docs.nvidia.com/tao/tao-toolkit/latest/text/vlm_finetuning/index.html) —
  dataset formats, Cosmos-Reason training and evaluation, and AutoML.

```{nvlearning-checkpoint} Course Complete
- You can describe the three-stage VSS alert verification pipeline and name the stage that controls false positives.
- You can say which Cosmos tower a given job needs, and why.
- You can run a zero-shot evaluation, read the result, and classify the gap it reveals.
- You can generate targeted synthetic data with EVG and VDA and verify it before training on it.
- You can fine-tune Cosmos 3 Reasoner with LoRA through Agent Skills, merge the adapter, and redeploy it into VSS.
- You can explain why model-level and system-level metrics differ and report each correctly.
- You can explain what each blueprint adds on top of calling the model directly.
```

```{nvlearning-feedback}
Tell us what worked and what didn't. Your feedback shapes the next revision of
this course.
```
