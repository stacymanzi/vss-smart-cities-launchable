#!/usr/bin/env python3
"""Assemble the lab dataset into the layout the notebook consumes.

    dataset/
      train/videos/*.mp4
      train/annotations_original.json     originals only
      train/annotations_augmented.json    originals + weather/lighting variants
      val/videos/*.mp4
      val/annotations_original.json
      val/annotations_augmented.json
      dataset_info.json                   task metadata + counts

Sources are the three archives published for this lab. Their on-disk shapes
differ — the augmented TRAIN clips sit in class folders with the condition in the
filename, while the augmented VAL clips sit in condition folders — so both are
normalised here rather than in the notebook. Condition names are unified to
{night, rain, fog}.

Videos are hardlinked when possible, so the assembled dataset costs no extra disk
on the same filesystem.
"""
import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- the task this dataset encodes --------------------------------------
TASK = {
    "name": "traffic anomaly detection",
    "classes": ["collision", "stalled"],
    "negative": "traffic",
    "system": ("You are monitoring a fixed camera. The camera never moves. "
               "You classify each clip into exactly one label."),
    "question": ("Which of the following does this clip show: "
                 "a collision between vehicles, a stalled vehicle blocking a travel lane, "
                 "or neither? Reply with exactly one line: LABEL=<collision|stalled|none>"),
}
LABEL_OF = {**{c: c for c in TASK["classes"]}, TASK["negative"]: "none"}
ALL_DIRS = TASK["classes"] + [TASK["negative"]]

# augmentation vocabularies differ between the two source sets
CONDITION_ALIASES = {"nighttime": "night", "night": "night",
                     "heavyrain": "rain", "rain": "rain",
                     "fog": "fog"}


def record(clip_id, label, variant, source_clip):
    return {
        "id": clip_id,
        "video_id": f"{clip_id}.mp4",
        "video": f"videos/{clip_id}.mp4",
        "source_clip": source_clip,
        "label": label,
        "variant": variant,
        "conversations": [
            {"from": "human", "value": f"<video>\n{TASK['question']}"},
            {"from": "gpt", "value": f"LABEL={label}"},
        ],
    }


def place(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def collect_originals(root: Path):
    """<root>/<class>/<clip>.mp4 -> records."""
    out = []
    for cls in ALL_DIRS:
        for clip in sorted((root / cls).glob("*.mp4")):
            out.append((clip, record(clip.stem, LABEL_OF[cls], "original", clip.stem)))
    return out


def collect_aug_by_class(root: Path):
    """<root>/<class>/<clip>_<condition>.mp4 -> records (augmented TRAIN layout)."""
    out = []
    for cls in ALL_DIRS:
        for clip in sorted((root / cls).glob("*.mp4")):
            base, _, raw = clip.stem.rpartition("_")
            cond = CONDITION_ALIASES.get(raw)
            if not cond:
                print(f"  ! skipping {clip.name}: unrecognised condition {raw!r}")
                continue
            cid = f"{base}_{cond}"
            out.append((clip, record(cid, LABEL_OF[cls], cond, base)))
    return out


def collect_aug_by_condition(root: Path):
    """<root>/<condition>/<clip>.mp4 -> records (augmented VAL layout)."""
    out = []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        cond = CONDITION_ALIASES.get(cdir.name)
        if not cond:
            continue
        for clip in sorted(cdir.glob("*.mp4")):
            cls = clip.stem.rsplit("_", 1)[0]      # collision_01 -> collision
            if cls not in ALL_DIRS:
                continue
            cid = f"{clip.stem}_{cond}"
            out.append((clip, record(cid, LABEL_OF[cls], cond, clip.stem)))
    return out


def write_split(name, originals, augmented, out_root: Path):
    split = out_root / name
    vids = split / "videos"
    vids.mkdir(parents=True, exist_ok=True)

    for src, rec in originals + augmented:
        place(src, vids / f"{rec['id']}.mp4")

    orig_recs = [r for _, r in originals]
    # The augmented file is a SUPERSET: originals plus their variants. Training on
    # it means every source clip appears in both its clean and augmented form.
    aug_recs = orig_recs + [r for _, r in augmented]

    (split / "annotations_original.json").write_text(json.dumps(orig_recs, indent=2))
    (split / "annotations_augmented.json").write_text(json.dumps(aug_recs, indent=2))

    print(f"\n{name}/")
    print(f"  videos                      {len(list(vids.glob('*.mp4'))):>4}")
    print(f"  annotations_original.json   {len(orig_recs):>4}  {dict(sorted(Counter(r['label'] for r in orig_recs).items()))}")
    print(f"  annotations_augmented.json  {len(aug_recs):>4}  {dict(sorted(Counter(r['label'] for r in aug_recs).items()))}")
    print(f"  variants                         {dict(sorted(Counter(r['variant'] for r in aug_recs).items()))}")
    return orig_recs, aug_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips",     default=ROOT / "data_cosmos_super/traffic_anomaly_dataset")
    ap.add_argument("--aug-train", default=ROOT / "data_augmented")
    ap.add_argument("--aug-val",   default=ROOT / "data_augmented_eval")
    ap.add_argument("--out",       default=ROOT / "dataset")
    a = ap.parse_args()
    clips, out = Path(a.clips), Path(a.out)

    print(f"assembling {out}")
    tr_o, tr_a = write_split("train",
                             collect_originals(clips / "train_clips"),
                             collect_aug_by_class(Path(a.aug_train)), out)
    va_o, va_a = write_split("val",
                             collect_originals(clips / "eval_clips"),
                             collect_aug_by_condition(Path(a.aug_val)), out)

    # No source clip may appear in both splits — the whole point of the split.
    leak = {r["source_clip"] for r in tr_a} & {r["source_clip"] for r in va_a}
    assert not leak, f"source clips in both splits: {sorted(leak)[:5]}"

    (out / "dataset_info.json").write_text(json.dumps({
        "task": TASK,
        "label_of": LABEL_OF,
        "counts": {"train": {"original": len(tr_o), "augmented": len(tr_a)},
                   "val": {"original": len(va_o), "augmented": len(va_a)}},
        "conditions": sorted(set(CONDITION_ALIASES.values())),
    }, indent=2))
    print(f"\nno source clip spans train and val — OK")
    print(f"wrote {out}/dataset_info.json")


if __name__ == "__main__":
    main()
