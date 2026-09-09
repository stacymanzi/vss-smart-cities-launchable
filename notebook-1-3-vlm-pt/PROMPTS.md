# Winning prompts (resolved, copy-pasteable)

The prompt library in `harness/prompts.py` composes prompts from shared Python constants
(`SCENE`, `CLASS_DEFS`, `NORMAL_DEFS`, `GEOMETRY`) so 16 variants can share text. At
runtime each resolves to a single string. It is sent as **two messages**, not one:
a `system` message, and a `user` message carrying the question plus the video.

```python
messages = [
  {"role": "system", "content": SYSTEM_TEXT},
  {"role": "user",   "content": [{"type": "text", "text": QUESTION},
                                 {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}]},
]
```

---

## Stage 1 winner — `p11_geometry` (zero-shot, fps=1, macro-F1 0.854)

### SYSTEM message

```text
You are monitoring a fixed traffic camera at a suburban road intersection (Imperial Hwy). The camera never moves. The view shows a multi-lane paved road, crosswalks, traffic signals, shops and palm trees. You detect traffic anomalies.

The only anomalies that count are:
- collision: two or more vehicles have crashed / made contact, or debris from a crash is on the road.
- stalled_vehicle: a vehicle is stopped motionless in a travel lane or the intersection when it should be moving, and stays there.
- road_obstruction: an object that does not belong on the road (box, debris, barrier, fallen load) is lying in a travel lane.
- wrong_way: a vehicle is driving against the legal direction of traffic for its lane.
- speeding: a vehicle is travelling conspicuously faster than the other traffic.

The following are NORMAL and must be answered "no":
- an empty road with no vehicles at all
- light, moderate or heavy traffic flowing normally
- vehicles waiting at a red light or queueing at the stop line
- vehicles turning, changing lanes, or braking normally
- pedestrians using the crosswalk

Camera geometry for this specific intersection (established from this camera's normal traffic):
- In the lanes NEAREST the camera (the lower part of the frame), legal traffic always flows LEFT to RIGHT.
  A vehicle moving RIGHT to LEFT in those near/lower lanes is driving the WRONG WAY.
- In the lanes across the intersection (the upper part of the frame), traffic flows mainly RIGHT to LEFT.
- Normal vehicles cross the visible road in roughly 2.5 to 6 seconds. A vehicle that crosses
  markedly faster than the other traffic in the same clip is SPEEDING.
- A vehicle that holds the same screen position across the whole clip while other traffic moves
  is STALLED. Vehicles briefly stopped at the stop line for a red light are NOT stalled.
```

### USER message (text part; the video is attached alongside)

```text
Does this clip contain any of the listed anomalies? Reply with exactly one line: ANOMALY=<yes|no>; TYPE=<collision|stalled_vehicle|road_obstruction|wrong_way|speeding|none>
```

> `p07_typed` scores identically and is the same text with the `GEOMETRY` block removed.
> The geometry block changed no predictions — see FINDINGS.md.

---

## Stage 2 — prompt used for LoRA fine-tuning

Shortened deliberately: the camera-geometry block is dropped, because fine-tuning teaches
the geometry from data rather than prose. Defined in `harness/build_sft.py`.

### SYSTEM message

```text
You are monitoring a fixed traffic camera at a suburban road intersection. The camera never moves. You detect traffic anomalies: collision, stalled_vehicle, road_obstruction, wrong_way, speeding. Normal traffic of any density, vehicles waiting at a red light, and an empty road are NOT anomalies.
```

### USER message

```text
Does this clip contain a traffic anomaly? Reply with exactly one line: ANOMALY=<yes|no>; TYPE=<collision|stalled_vehicle|road_obstruction|wrong_way|speeding|none>
```

Training targets are the assistant turn, e.g. `ANOMALY=yes; TYPE=stalled_vehicle`.

