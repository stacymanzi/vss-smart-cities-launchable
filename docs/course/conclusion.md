# Conclusion

You started with a stock VSS deployment that produced zero traffic candidates
and finished with a fine-tuned pipeline that recovers a collision the base
model missed, with no new false positives. Everything in between was one loop,
run once, with measurements at both ends.

```{nvlearning-meta}
- **Level:** Intermediate
- **Covers:** Course recap, VSS profiles, and where to go next
```

## What You Built

An end-to-end vision AI agent for smart-city traffic anomaly detection:

1. **RT-CV** detects and tracks vehicles with the DeepStream SDK.
2. **Behavior Analytics** converts tracks into candidate alerts for collisions
   and stalled vehicles.
3. **Alert Bridge** sends each candidate clip to Cosmos Reason 3 Nano for a
   confirm-or-reject verdict.

You changed exactly one component — the verifier — and measured what that was
worth.

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

## Where to Go Next

- [NVIDIA Blueprint for Video Search and Summarization](https://docs.nvidia.com/vss/latest/) —
  full reference architectures, microservice details, and deployment options.
- [NVIDIA TAO Toolkit VLM fine-tuning](https://docs.nvidia.com/tao/tao-toolkit/latest/text/vlm_finetuning/index.html) —
  dataset formats, Cosmos-Reason training and evaluation, and AutoML.
- [NVIDIA Physical AI Data Factory](https://github.com/NVIDIA/physical-ai-data-factory) —
  the EVG and VDA workflows, cookbooks, and auto-labeling modules.
- [NVIDIA Cosmos](https://github.com/nvidia/cosmos) — world foundation models,
  the Reasoner and Generator surfaces, and model sizes from Edge to Super.

```{nvlearning-checkpoint} Course Complete
- You can describe the three-stage VSS alert verification pipeline and name the stage that controls false positives.
- You can run a zero-shot evaluation, read the result, and classify the gap it reveals.
- You can generate targeted synthetic data with EVG and VDA and verify it before training on it.
- You can fine-tune Cosmos Reason with LoRA through Agent Skills, merge the adapter, and redeploy it into VSS.
- You can explain why model-level and system-level metrics differ and report each correctly.
```

```{nvlearning-feedback}
Tell us what worked and what didn't. Your feedback shapes the next revision of
this course.
```
