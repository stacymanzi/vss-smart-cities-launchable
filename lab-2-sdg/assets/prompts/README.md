# Lab 2 prompt files

All prompts describe a fixed traffic-camera view of a signalized intersection (the seed frame in
`assets/inputs/site03_median.jpg`). Wording is neutral traffic-operations language (traffic camera,
vehicle, pedestrians in the roadway); no enforcement or identification framing.

## Qwen edit instructions (`edit_examples.json`)

Example instructions for the seed image, listed by the 1.2 cell for inspiration. The notebook takes a free-text
`EDIT_PROMPT`; these are not a menu. The notebook's default instruction is `snowy_morning` (seed 42, Stacy's choice,
not yet generated on the devbox). `rainy_morning` (seed 42) is validated on the devbox; the others have not been generated there yet.

## I2V scenario presets (`I2V_PROMPT_FILES` in the notebook)

| File | Preset | Status |
|---|---|---|
| `i2v_normal_traffic.json` | `normal_traffic` | validated on the devbox (Sep 3, seed 0) - the default |
| `i2v_red_light_running.json` | `red_light_running` | not yet generated on the devbox |
| `i2v_wrong_way_driving.json` | `wrong_way_driving` | not yet generated on the devbox |
| `i2v_stalled_vehicle.json` | `stalled_vehicle` | new prompt following `i2v_prompt_schema.json`; not yet generated on the devbox |
| `i2v_collision.json` | `collision` | new prompt following `i2v_prompt_schema.json`; not yet generated on the devbox |

`i2v_prompt_schema.json` documents the schema; `llm_generated_prompt.json` is the example output of the
optional 2.2.1 step (shown for reference when the step is skipped). The notebook never writes into this
directory: generated prompts go to `assets/outputs/prompts/`.

## Transfer presets (`TRANSFER_PROMPT_FILES`)

| File | Preset | Status |
|---|---|---|
| `transfer_nighttime_snowy.json` | `nighttime_snowy` | validated on the devbox (Sep 3, seed 2026, 193 frames / 45 steps, guidance 3, control_guidance 2, shift 5) - the default |
| `transfer_nighttime_wet.json` | `nighttime_wet` | ran on the devbox (Sep 3) with earlier settings; not yet re-run with the final ones |

`transfer_prompt_schema.json` documents the schema; `llm_generated_transfer_prompt.json` is the example
output of the optional 3.1.1 step.

## Negative prompts (in `assets/`)

- `negative_prompt_i2v.json` - used by the I2V payload (`negative_prompt` inline string). Copy of the
  Cosmos cookbook file `cookbooks/cosmos3/generator/audiovisual/assets/negative_prompts/image2video/neg_prompt.json`.
- `negative_prompt.json` - used by the transfer spec (`negative_prompt_file`). Same cookbook source.
