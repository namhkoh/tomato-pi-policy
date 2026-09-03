# Qwen cut-point evaluation

This package evaluates an OpenAI-compatible vision model on one unannotated
greenhouse RGB frame. It asks for both:

- a tight bounding box around the selected petiole and attachment region, to
  reveal which structure the model selected; and
- one pixel cut point and image-plane blade direction, which are the quantities
  needed for later depth back-projection and motion planning.

The point is the primary task output. A bounding box alone is not precise enough
to score residual stub length or initialize a cutting pose.

## Run the supplied Qwen endpoint

Keep the API key out of the command line and repository. `data/` is gitignored,
so an ignored key file is one safe local option:

```powershell
$keyFile = "data\greenhouse_sim\vlm_eval\qwen_api_key.txt"
# Populate $keyFile locally without committing or printing the key.

D:\isaac-sim\python.bat examples\greenhouse_sim\vlm_eval\run_qwen_cutpoint.py `
  --base-url https://poster-tattoo-citations-billion.trycloudflare.com/v1 `
  --model qwen3-vl-32b-instruct `
  --api-key-file $keyFile `
  --image data\greenhouse_sim\evidence\latest_20260817_substem01_final\frames\000092_left_wrist.png `
  --view-name left_wrist
```

The trycloudflare URL is ephemeral. Replace `--base-url` after the tunnel is
restarted, or set `QWEN_BASE_URL`. `QWEN_API_KEY` can be used instead of an API
key file when the evaluator process inherits that variable.

The immutable Isaac PNG remains the scored source. By default the adapter
submits an in-memory quality-92 RGB JPEG with identical dimensions because this
particular vLLM endpoint returns HTTP 500 on the Isaac PNG and on the larger
quality-95 encoding. Both source and
submitted hashes plus the transcoding flag are recorded in `request.json`.

Each run writes the following ignored artifacts under
`data/greenhouse_sim/vlm_eval/runs/`:

- `request.json`: prompt, image dimensions/hash, and generation settings; never
  the API key or base64 image;
- `raw_response.json`: complete provider response;
- `prediction.json`: validated canonical proposal, latency, mode, and usage;
- `overlay.png`: petiole box, cut point, and blade direction for human review;
- `error.json`: retained endpoint or schema error when a trial fails.

## Interpretation

One plausible overlay is not benchmark evidence. The initial single-frame test
checks transport, schema compliance, basic plant-part reasoning, and coordinate
rendering. Quantitative claims require private simulator projections and masks,
negative/no-safe-cut examples, repeated trials, and plant-disjoint evaluation as
defined in `docs/vlm_cutpoint_evaluation.md`.
