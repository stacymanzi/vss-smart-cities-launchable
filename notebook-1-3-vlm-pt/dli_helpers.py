"""Helpers for the Singapore AI Day fine-tuning notebook.

Plumbing only — video cutting, the docker invocation, and scoring. The parts a
student should actually read (dataset design, prompt, hyperparameters, results)
stay inline in the notebook cells.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ----------------------------------------------------------------- constants

CLASSES = ["road_obstruction", "stalled_vehicle", "collision", "wrong_way", "speeding"]
IMAGE = "nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-rl"


# ----------------------------------------------------------------- preflight

def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    return ok


def gpu_table() -> list[dict]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        i, name, tot, used = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(i), "name": name,
                     "total_gb": round(int(tot) / 1024, 1),
                     "used_gb": round(int(used) / 1024, 1)})
    return gpus


# ------------------------------------------------------------------ dataset

def cut_window(args):
    src, dst, t0, dur = args
    if dst.exists() and dst.stat().st_size > 0:
        return
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(src),
         "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-an", str(dst), "-y"], check=True)


def windows(start: float, end: float, win: float, stride: float):
    """Dense overlapping windows; whole span when shorter than one window."""
    if end - start <= win:
        return [(start, end)]
    out, t = [], start
    while t + win <= end + 1e-6:
        out.append((t, t + win))
        t += stride
    if out and out[-1][1] < end - 1.0:
        out.append((end - win, end))
    return out


def cut_all(jobs, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(cut_window, jobs))


def write_llava(records, out_dir: Path):
    """Write the annotation the TAO cosmos-rl trainer reads.

    The container invokes /opt/cosmos_rl/tao_sft_example.py, which does
    `len(json.load(f))` — so the file must be a JSON **list**, and each item
    carries `video` plus a `conversations` pair of {from, value}. (A different
    hook in the same image expects {media_root, items:[...]}; feeding that shape
    here yields a silent zero-length dataset.)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations.json").write_text(json.dumps(records, indent=2))
    return out_dir / "annotations.json"


# -------------------------------------------------------------------- docker

def _docker_env(job_name: str, results_in_container="/results") -> list[str]:
    """Env every TAO cosmos-rl container needs when run as a non-root host uid.

    Each of these fixes a real failure:
      USER/LOGNAME        torch's cache_dir() calls getpass.getuser(); the host
                          uid has no /etc/passwd entry -> KeyError getpwuid
      TAO_API_JOB_ID      _get_results_dir() otherwise returns a RELATIVE
                          "./results" and fails with PermissionError
      *_CACHE_DIR/HOME    keep every cache inside the writable results mount
    """
    return [
        "-e", "USER=taouser", "-e", "LOGNAME=taouser", "-e", f"HOME={results_in_container}",
        "-e", f"TAO_API_JOB_ID={job_name}", "-e", f"TAO_API_RESULTS_DIR={results_in_container}",
        "-e", f"TORCHINDUCTOR_CACHE_DIR={results_in_container}/.inductor",
        "-e", f"TRITON_CACHE_DIR={results_in_container}/.triton",
        "-e", f"XDG_CACHE_HOME={results_in_container}/.cache",
        "-e", f"HF_HOME={results_in_container}/.hf",
    ]


# Container logs are extremely noisy (vLLM engine banners, Triton/pyarmor import
# warnings). Show only lines a student should read.
_KEEP = re.compile(r"avg_loss|Validation at step|Exported safetensors|Checkpoint will be saved"
                   r"|checkpoint saved|Job FAILED|Traceback|AssertionError|Error:", re.I)
_DROP = re.compile(r"pyarmor|Triton kernels|Telemetry|EngineCore|WARNING|UserWarning"
                   r"|Failed to import|deprecat", re.I)


def _interesting(line: str) -> bool:
    return bool(_KEEP.search(line)) and not _DROP.search(line)


def run_container(job_name, gpu, mounts, command, log_path, timeout=7200, stream=True):
    """Run one TAO container, streaming a filtered view of its log."""
    cmd = ["docker", "run", "--rm", "--name", job_name,
           "--gpus", f'"device={gpu}"', "--shm-size=32g", "--ipc=host",
           "-u", f"{os.getuid()}:{os.getgid()}", "-w", "/results"]
    cmd += _docker_env(job_name)
    for host, cont in mounts:
        cmd += ["-v", f"{host}:{cont}"]
    cmd += [IMAGE] + command

    subprocess.run(["docker", "rm", "-f", job_name], capture_output=True)
    t0 = time.time()
    with open(log_path, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        last = 0
        while p.poll() is None:
            time.sleep(5)
            if stream:
                lines = Path(log_path).read_text(errors="ignore").splitlines()
                for l in lines[last:]:
                    if _interesting(l):
                        print("   ", l[:150], flush=True)
                last = len(lines)
    dt = time.time() - t0
    print(f"\n[{job_name}] exit={p.returncode} in {dt/60:.1f} min  (log: {log_path})")
    return p.returncode, dt


# ------------------------------------------------------------------- scoring

def parse_answer(text: str):
    """'ANOMALY=yes; TYPE=collision' -> ('yes', 'collision')."""
    t = (text or "").lower()
    m = re.search(r"anomaly\s*=\s*(yes|no)", t)
    binary = m.group(1) if m else ("yes" if "anomaly=yes" in t else "no")
    cls = "none"
    mm = re.search(r"type\s*=\s*([a-z_]+)", t)
    if mm:
        for c in CLASSES:
            if c in mm.group(1):
                cls = c
                break
    return binary, cls


def score(items):
    """items: [{'answer','pred','anomaly_class'}] -> metrics dict.

    macro-F1 averages the anomaly-class and normal-class F1 equally, so a model
    that just answers "no" to everything cannot score well on an imbalanced set.
    """
    tp = sum(1 for i in items if i["answer"] == "yes" and i["pred"] == "yes")
    fp = sum(1 for i in items if i["answer"] == "no" and i["pred"] == "yes")
    fn = sum(1 for i in items if i["answer"] == "yes" and i["pred"] == "no")
    tn = sum(1 for i in items if i["answer"] == "no" and i["pred"] == "no")

    def f1(t, f_p, f_n):
        pr = t / (t + f_p) if (t + f_p) else 0.0
        rc = t / (t + f_n) if (t + f_n) else 0.0
        return (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0, pr, rc

    f1_pos, prec, rec = f1(tp, fp, fn)
    f1_neg, _, _ = f1(tn, fn, fp)

    per = defaultdict(lambda: [0, 0])
    for i in items:
        if i["answer"] == "yes":
            per[i["anomaly_class"]][1] += 1
            per[i["anomaly_class"]][0] += i["pred"] == "yes"

    return {"macro_f1": (f1_pos + f1_neg) / 2, "precision": prec, "recall": rec,
            "accuracy": (tp + tn) / len(items) if items else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "per_class": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per.items())}}


def show(name, m):
    print(f"{name:<28} macro-F1={m['macro_f1']:.3f}  P={m['precision']:.3f}  "
          f"R={m['recall']:.3f}  acc={m['accuracy']:.3f}")
    print(f"{'':<28} per-class detected: {m['per_class']}")


def load_eval_results(results_dir: Path):
    """Read the evaluator's per-sample predictions and score them."""
    files = list(Path(results_dir).rglob("results.json"))
    if not files:
        raise FileNotFoundError(f"no results.json under {results_dir}")
    data = json.loads(files[0].read_text())
    items = []
    for r in data:
        pred, _ = parse_answer(r.get("response") or "")
        gold, gcls = parse_answer(r.get("gt") or "")
        items.append({"pred": pred, "answer": gold, "anomaly_class": gcls})
    return items, files[0]
