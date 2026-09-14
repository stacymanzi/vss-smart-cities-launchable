# Lab 1 media (course-owned assets only)

The notebook reads two files from `LAB_DATA_DIR` (default `/dli/task/data`). They are staged at learner launch by the course
data loader from course storage; the notebook never downloads anything.

| File | What it is | Constraints |
|---|---|---|
| `traffic.png` | One frame from a fixed traffic-camera view | PNG, up to 1920x1080 |
| `traffic_short.mp4` | One 8-30 s traffic-camera clip containing a single incident (stalled vehicle or collision) | H.264 MP4, <= 720p, <= 15 MB (embedded inline in the notebook preview) |

Keep the clip short and at most 1280x720: at the NIM default of 4 FPS sampling, a 30 s 720p clip is
`30 s x 4 fps / 2 x (720 x 1280 / 1024) = 54,000` vision tokens, well above the ~16k guidance. An 8 s 720p clip is
~14.4k tokens; a 30 s clip should be 540x960 or smaller (~24k at 30 s, ~16k at 20 s).

## Provenance

All media in this lab is **course-owned and synthetic**. It is derived from the traffic-anomaly dataset generated with
NVIDIA Cosmos (Cosmos 3 Super, image-to-video) for this course - the same dataset Lab 2 and Part 2 (VSS alert
verification) use - whose seed frame was derived from the CalTrans camera feed cleared under DGPTT-5636.

| File | Generator | Source clip / seed | License / clearance | Prepared by / date |
|---|---|---|---|---|
| `traffic.png` | Frame extracted from `traffic_short.mp4` at 5 s (`ffmpeg -ss 00:00:05 -i traffic_short.mp4 -frames:v 1 traffic.png`) | `traffic_short.mp4` | Course-owned Cosmos output | Hassan Moustafa, Sep 2 2026 |
| `traffic_short.mp4` | Cosmos 3 Super image-to-video, from the course traffic-anomaly eval set (collision clip `collides.mp4`, trimmed to the first 16 s) | course eval set, collision clip | Course-owned Cosmos output | Hassan Moustafa, Sep 2 2026 |

There are no real people or licence plates in the media. No third-party dataset footage of any kind (public benchmark sets, third-party generative
models or any other source) may be used in this lab.

## Staging

Staged by the course loader (`task1/data_sources`) from `s3://dli-lms/data/x-fx-138-v1/` into the named data volume mounted
at `/dli/task/data/`. The two files are listed before the recursive entry so they land first; the Cosmos 3 Reasoner NIM cache
(`cosmos3-reasoner-cache/`, ~9.9 GB) follows.
