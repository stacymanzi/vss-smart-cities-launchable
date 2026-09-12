# Part 2.1: Deploy Zero-Shot VSS for Alert Verification

VSS Alert Verification is a three-stage pipeline. RT-CV detects and tracks
vehicles, Behavior Analytics turns tracks into candidate alerts (stopped
vehicle, collision), and Alert Bridge sends each candidate clip to Cosmos
Reason 3 Nano, which confirms or rejects it.

In this walkthrough you deploy that pipeline with the VSS agent skills,
discover why a stock deployment is only the starting point for a traffic
camera, then apply the traffic-tuned profile and watch real alerts arrive.

```{nvlearning-meta}
- **Level:** All levels — no prior VLM or Cosmos experience needed
- **Duration:** About 45 minutes, dominated by two deployment waits
- **Agent:** Codex with the VSS skills
- **Working directory:** `/opt/vss`
```

## Learning Objectives

```{nvlearning-objectives}
- **Set up** the coding agent and install the VSS skills.
- **Deploy** the stock alerts profile in verification mode with `vss-deploy-profile`.
- **Explain** what each deployed component does while the stack comes up.
- **Add** an evaluation stream and an alert rule with the operate skills, and interpret the result.
- **Apply** the traffic-tuned CV profile with stock Cosmos Reason 3 Nano and run an evaluation video.
- **Read** the alerts and Cosmos verdicts in the UI.
```

## Step 1: Log the Agent In

Open a terminal in JupyterLab. The device flow prints a URL and a code — open
the URL in your browser, sign in, and enter the code.

```bash
codex login --device-auth
cd /opt/vss && codex
```

## Step 2: Install the Skills

Paste into Codex:

```text
Read /opt/vss/skills/README.md and install those skills for this host
so I can invoke them from a shell or chat session.
```

## Step 3: Give the Agent the Environment Context

Paste this as **one message** before any deploy. It is the difference between
a deployment that works and an hour of debugging the wrong namespace.

```text
Context before you begin. This is a Docker-in-Docker environment, so
several things differ from a workstation:

1. The Docker daemon is NOT local. It is remote, reached via DOCKER_HOST,
   and is already configured for VSS. Verify it with `docker info` rather
   than reading /etc/docker/daemon.json or running sysctl locally - those
   describe this container, not the daemon. There is no sudo here and
   none is needed.

2. HOST_IP is set to "dind" in this shell. That is for PROBING services
   from here only. Do NOT pass it as --host-ip, and do NOT derive the
   address from this container's own interfaces or routes (`ip route get`,
   `hostname -I`). The only correct value is the daemon's own address:

     docker run --rm --network host busybox:1.37.0 \
       ip -4 addr show eth0 | awk '/inet /{print $2}' | cut -d/ -f1

   (equivalently: getent hosts dind). Pass that IP as --host-ip. It
   changes whenever the daemon container is recreated, so detect it each
   time rather than reusing a previous value.

3. ALWAYS pass --external-ip explicitly, even though it is optional, and
   ASK ME for the value before you deploy. Do not derive it, and never
   reuse --host-ip for it. If you omit it, EXTERNAL_IP silently becomes
   HOST_IP, every container is healthy, and the UI is unreachable from any
   real browser with no error anywhere.

4. The net.core.rmem_max / wmem_max sysctl warning is expected and safe
   to overrule. They are not network-namespaced and cannot be set from
   inside a container. Deployments succeed with the current values.

5. Everything you can see through DOCKER_HOST is a warm cache that is
   expensive to rebuild: ~33 images (about 90 GB) plus model weight
   volumes. `dev-profile.sh down` is fine, but never pass -v to a compose
   down and never run `docker system prune`.

6. An idle or empty component is usually correct, not broken. RT-CV is a
   stream-processing worker, not an HTTP server, and processes zero sources
   until a stream is registered in VST. The Alert UI stays empty until an alert
   rule exists. Check `/vst/api/v1/sensor/list` and the alert-bridge rule
   list before concluding something is misconfigured.

When something looks wrong here, ask two questions before changing any
setting: which namespace am I standing in, and has this component been
given work?
```

## Step 4: Deploy the Stock Profile

Paste into Codex. When it asks for the external address, give it the hostname
from your browser's address bar, **without** `/lab`.

```text
Using the vss-deploy-profile skill, deploy the alerts profile in
verification mode on H100, with the LLM and VLM sharing GPU 1.
Before deploying, show me the exact HOST_IP and EXTERNAL_IP values you
will use and how you obtained each. Explain what each component does
as it comes up.
```

:::{note}
**Expected:** a plan within about 4 minutes listing about 31 services, then all
healthy in about 17 minutes.
:::

:::{only} internal
The instructor covers the architecture during this 17-minute window.
:::

## Step 5: Watch It Come Up

In a second terminal:

```bash
watch -n 15 'docker ps --format "table {{.Names}}\t{{.Status}}" | sort'
```

Expected order: `elasticsearch`, `kafka`, `redis` and `postgres` first; then
VIOS (ingress, sensor, streamprocessing); then RT-VLM and the Nemotron NIM
healthy; finally `agent`, `alert-bridge` and RT-CV.

RT-CV exposes its API on port **9010** in this environment, not 9000.

## Step 6: Open the UI

Browse to `http://<your-lab-host>/alerts/`.

The Alerts tab is empty. That is correct — no rule exists yet.

## Step 7: Give the System Work

Paste into Codex:

```text
Using the vss-manage-video-io-storage and vss-manage-alerts skills:
upload /dli/task/data/traffic_anomaly_dataset/eval_videos/eval_video_01.mp4
to VIOS as a stream named traffic-eval-01, attach it to RT-CV, create a
verification alert rule for vehicle collisions and stopped vehicles on
that stream, then wait for the clip to finish and show me the resulting
candidates, incidents and Cosmos verdicts.
```

**Expected (measured):** the stream registers and plays (3-minute clip), then
produces **0 candidates and 0 verdicts**. The agent reports that the profile
only enables the warehouse ladder/PPE rule, and that the stock stopped-vehicle
rule needs 120 seconds of stillness.

This is not a failure. It is the motivation for the tuned profile: a stock
deployment ships with rules written for a warehouse, and a traffic camera
needs different thresholds and a different CV configuration.

## Step 8: Prepare the Tuned-Profile Build

Back in the JupyterLab terminal, before launching Codex again. The NGC key is
typed silently and lives only in this shell.

```bash
export AICITY_PUBLIC_URL="http://<your-lab-host>"
export AICITY_MEDIA_DIR=/dli/task/data
export AICITY_DATA_DIR=/opt/aicity-runtime/aicity-vss-data
export AICITY_BUILD_DIR=/opt/vss/_builds/aicity-track4-alert-verification-dli-cvfix4
export NGC_CLI_ORG=nvidia
read -rsp 'NGC Personal Key: ' NGC_CLI_API_KEY && echo && export NGC_CLI_API_KEY NGC_API_KEY="$NGC_CLI_API_KEY"
printf '%s\n' "$NGC_CLI_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
cd /opt/vss && codex
```

## Step 9: Apply the Tuned Traffic Profile

Paste into Codex:

```text
Read /dli/task/aicity_cvfix4/README.md and apply the tuned AI City traffic
profile on this machine, following its sections in order through section 7
(status and ready). The AICITY_* variables, HOST_IP and NGC_CLI_API_KEY are
already exported in this shell. The stock VSS alerts deployment from earlier
is still running: stop it without -v first, and stash any tracked changes in
/opt/vss so the installer accepts the checkout. Ask me before any destructive
step and report the exact commands you ran at the end.
```

**Expected:** `install.sh` verifies the package and pins `/opt/vss` to the
experiment commit; the unit suite passes; structural validation resolves 19
services; preflight passes (the sysctl warning is expected); `up` ends with
"All AI City VSS services, UI, and ingress capability probes are ready."
About 10 minutes including the Cosmos load.

Answer **yes** when the agent asks to remove a stale VIOS symlink.

## Step 10: Run an Evaluation Video

Paste into Codex:

```text
Run README section 8 with
/dli/task/data/traffic_anomaly_dataset/eval_videos/eval_video_01.mp4,
run id stock-eval-01 under /opt/aicity-runtime/aicity-eval,
--vlm-model nim_nvidia_cosmos3-nano-reasoner_bf16-final, keeping
--retain-vios-storage. Show me the run manifest counts and the Cosmos
verdicts.
```

**Expected in about 6 minutes:** `failure: null`, nonzero
`rt_cv_detection_frames`, candidates for the two collisions and two stalled
vehicles in the clip, and a Cosmos verdict per candidate.

Refresh the Alerts tab. Rows appear under sensor `stock-eval-01` with alert
type `collision` or `Stop Anomaly Module`, and verdict `Confirmed` or
`Rejected`. Click a row to see the clip with the green target overlay.

```{nvlearning-checkpoint} Checkpoint 1
- Codex logged in, VSS skills installed.
- Stock alerts profile deployed, all probes healthy, UI reachable at `/alerts/`.
- Evaluation stream registered and alert rule created; agent reported zero traffic candidates and explained why.
- Tuned traffic profile deployed with the stock Cosmos Reason 3 Nano; `eval_video_01` produced candidates and verdicts visible in the UI.
```

## Troubleshooting

| It reports | What is actually true |
|---|---|
| HAProxy cannot resolve `dind`, or services unhealthy after deploy | `--host-ip` was `dind` or this container's own address. Redeploy with the daemon address from the busybox probe. |
| Deploy succeeded but the UI 404s | `--external-ip` was omitted or wrong. Redeploy with the browser hostname. |
| RT-CV has zero sources; a port 9000 probe fails | No stream is registered yet, and RT-CV listens on 9010 here. |
| Alerts tab is empty | No rule exists. Registering a stream is not enough. |
| `install.sh`: "has tracked changes; use a fresh course deployment" | The stock deployment edited `/opt/vss`. Run `git -C /opt/vss stash` and rerun; the agent does this when told the checkout may be dirty. |
| `build.py`: "dangling symlink under VSS_DATA_DIR" | Leftover VIOS temp link from a previous run. Delete the link and rerun. |

:::{only} internal
**Shot list.**

- Codex device-login prompt in the JupyterLab terminal.
- Codex's deployment plan (31 services, GPU 0 and GPU 1 assignment) before "deploy now?".
- `watch docker ps` with all stock services healthy.
- Alerts UI empty after the stock deploy.
- Agent output: stream registered, "0 candidates" with its explanation.
- Codex summary "AI City tuned traffic profile is deployed and ready" (19 services).
- Alerts UI with `stock-eval-01` rows: collision Confirmed, Stop Anomaly Module Confirmed/Rejected.
- Clip viewer with the green target overlay on a stopped vehicle.
:::

## What's Next

You have a working traffic pipeline and a baseline. The stock Cosmos Reason 3
Nano confirms most real events, but not all of them. Before you fine-tune it,
you need data that targets the gap — which is what
[Part 2.2: Expanding Your Dataset With PAIDF](part-2-2-expand-your-dataset-with-paidf)
generates.
