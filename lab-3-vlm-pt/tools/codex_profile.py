#!/usr/bin/env python3
"""Per-turn latency profile of a Codex session transcript.
usage: codex_profile.py rollout.jsonl [--cmds]   (--cmds lists every command with its duration)"""
import json, sys, re, collections
from datetime import datetime
def ts(s): return datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()
def classify(cmd):
    c = cmd.strip()
    if re.search(r"docker (run|start)\b", c): return "docker run"
    if re.search(r"docker (logs|wait|inspect|ps|stop|rm)\b", c): return "docker watch"
    if re.search(r"\bsleep\b", c): return "sleep/poll"
    if re.search(r"tao_job_record|check_tao_launch_preflight|redact_secrets|render-docker|cosmos_workflow", c): return "skill scripts"
    if re.search(r"\bnvidia-smi\b|docker (info|images|image inspect)|df -h|id -u|free -g", c): return "env probes"
    if re.search(r"\b(cat|sed -n|head|tail|less|grep|rg|find|ls|wc|tree|stat)\b", c) and "skill" in c: return "read skill files"
    if re.search(r"\b(cat|sed -n|head|tail|grep|rg|find|ls|wc|stat|md5sum)\b", c): return "read other files"
    if re.search(r"python3? -c|python3? -\b|python3? <<|\.py\b", c): return "python (own scripts)"
    if re.search(r"\b(mkdir|cp|mv|tee|printf .*>|echo .*>|chmod|ln)\b", c): return "write/setup files"
    return "other"
def main(path, show_cmds=False):
    ev = []
    for l in open(path):
        l = l.strip()
        if not l: continue
        try: ev.append(json.loads(l))
        except json.JSONDecodeError: pass
    turns = []; cur = None; pending = {}
    for e in ev:
        t, p = e.get("type"), e.get("payload", {}) or {}
        pt = p.get("type"); T = ts(e["timestamp"])
        if (t == "event_msg" and pt == "user_message") or (t == "response_item" and pt == "message" and p.get("role") == "user"):
            txt = p.get("message") if pt == "user_message" else " ".join(c.get("text","") for c in p.get("content",[]) if isinstance(c,dict))
            if txt.lstrip().startswith("<") and pt != "user_message": continue   # injected context, not a prompt
            cur = {"prompt": txt[:70].replace("\n"," "), "t0": T, "t1": None, "cmds": [], "patches": 0, "msgs": 0, "reason": 0, "tokens": None}
            turns.append(cur); continue
        if cur is None: continue
        if t == "event_msg" and pt == "task_complete": cur["t1"] = T
        elif t == "response_item" and pt in ("function_call", "custom_tool_call"):
            if pt == "custom_tool_call":
                pending[p.get("call_id")] = (T, p.get("name"), str(p.get("input") or "")); continue
            try: a = json.loads(p.get("arguments") or "{}")
            except Exception: a = {}
            cmd = a.get("cmd") or a.get("command") or ""
            if isinstance(cmd, list): cmd = " ".join(cmd)
            pending[p.get("call_id")] = (T, p.get("name"), cmd)
        elif t == "response_item" and pt in ("function_call_output", "custom_tool_call_output"):
            k = p.get("call_id")
            if k in pending:
                T0, name, cmd = pending.pop(k)
                out = p.get("output") or ""
                if isinstance(out, list): out = json.dumps(out)
                cur["cmds"].append({"t0": T0, "dur": T - T0, "name": name, "cmd": cmd, "cls": classify(cmd) if name in ("exec_command","shell","shell_command","container.exec") else (name or "tool"), "out": len(out)})
        elif t == "event_msg" and pt == "agent_message": cur["msgs"] += 1
        elif t == "response_item" and pt == "reasoning": cur["reason"] += 1
        elif t == "event_msg" and pt == "token_count":
            u = (p.get("info") or {}).get("total_token_usage") or {}
            if u: cur["tokens"] = u
    print(f"{'turn':<4}{'wall':>7}{'cmd-time':>9}{'model':>7}{'calls':>6}{'reason':>7} prompt")
    for i, tr in enumerate(turns, 1):
        if tr["t1"] is None: tr["t1"] = max([c["t0"]+c["dur"] for c in tr["cmds"]] + [tr["t0"]])
        wall = tr["t1"] - tr["t0"]; ct = sum(c["dur"] for c in tr["cmds"])
        print(f"{i:<4}{wall/60:>6.1f}m{ct/60:>8.1f}m{(wall-ct)/60:>6.1f}m{len(tr['cmds']):>6}{tr['reason']:>7} {tr['prompt']}")
        by = collections.defaultdict(lambda: [0, 0.0])
        for c in tr["cmds"]: by[c["cls"]][0] += 1; by[c["cls"]][1] += c["dur"]
        for k, (n, d) in sorted(by.items(), key=lambda x: -x[1][1]): print(f"        {n:>3} x {k:<20} {d/60:>5.1f}m")
        if tr["tokens"]: u = tr["tokens"]; print(f"        tokens: in={u.get('input_tokens')} cached={u.get('cached_input_tokens')} out={u.get('output_tokens')} reasoning={u.get('reasoning_output_tokens')}")
        if show_cmds:
            for c in sorted(tr["cmds"], key=lambda c: -c["dur"])[:12]: print(f"          {c['dur']:>6.1f}s {c['cls']:<18} {c['cmd'][:110].replace(chr(10),' ')}")
if __name__ == "__main__": main(sys.argv[1], "--cmds" in sys.argv)
