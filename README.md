# LLM Council (Local NIM/vLLM)

![llmcouncil](header.jpg)

Backend-only LLM council that runs local NVIDIA NIM containers (vLLM backend), supports multi-round deliberation, and now captures structured prediction evidence for explainability and counterfactual reruns.

## What Changed

- Local NIM/OpenAI-compatible inference (no OpenRouter).
- Configurable N-round council deliberation with dynamic early stopping.
- Structured prediction + evidence extraction (3-5 evidence items) for every model response and synthesis.
- Observer-chairman trajectory: chairman synthesis is saved after every round.
- Evidence index endpoint for research workflows.
- Counterfactual endpoint to mask selected evidence spans and rerun the full council.

## Configuration

### 1. Create environment file

```bash
cp .env.example .env
```

Set:

- `NGC_API_KEY` (required for NIM image/model pulls)
- `DATA_DIR` (optional, defaults to `/data/results`)
- `MODELS_YAML` (optional, defaults to `./models.yaml`)
- `MODEL_ENDPOINT_HOST` (optional, defaults to `localhost`; set to a Docker DNS host when backend is containerized)
- `MODEL_ENDPOINT_SCHEME` (optional, defaults to `http`)
- `MODEL_ENDPOINT_<MODEL_NAME>` (optional per-model endpoint override, e.g. `MODEL_ENDPOINT_QWEN2_5_72B_INSTRUCT`)

### 2. Edit `models.yaml`

`models.yaml` is the source of truth for:

- round count (`rounds`)
- model names
- request model IDs (`request_model`, optional)
- extractor model (`extractor_model`, optional, defaults to chairman)
- deliberation controls (`deliberation.*`) for observer mode + early stopping
  - `observer_chairman`: chairman synthesis trajectory is tracked each round
  - `share_synthesis_with_members`: whether that synthesis is shown to members in later rounds
- stage-specific inference controls (`inference.round1`, `inference.round_n`, `inference.synthesis`, `inference.extractor`)
- NIM images
- GPU assignments
- ports
- chairman selection

## Generate Compose

```bash
python generate_compose.py
```

This writes `docker-compose.yml` from `models.yaml`.
Compose generation uses the same Pydantic schema validation as the backend loader.

## Run

```bash
uv sync
./start.sh
```

`start.sh`:

1. runs `docker compose up -d`
2. starts FastAPI on `http://localhost:8001`
3. waits until `/api/health` reports model readiness (including configured `request_model` availability)

## API Endpoints

- `POST /api/deliberate`
- `POST /api/batch`
- `POST /api/counterfactual`
- `GET /api/batch/{batch_id}`
- `GET /api/evidence/{batch_id}/{prompt_id}`
- `GET /api/models`
- `GET /api/health`

### Deliberate Request

```json
{
  "prompt": "optional extra instruction",
  "trial_text": "optional trial textblock",
  "prediction_target": "duration|price|success|...",
  "rounds": 3,
  "allow_fuzzy_quotes": false,
  "early_stopping": true,
  "min_rounds_before_stop": 2,
  "share_synthesis_with_members": false,
  "inference": {
    "round1": {"temperature": 0.7, "max_tokens": 2200},
    "round_n": {"temperature": 0.5, "max_tokens": 2200},
    "synthesis": {"temperature": 0.25, "max_tokens": 1800}
  },
  "metadata": {"study_id": "NCT..."}
}
```

At least one of `prompt` or `trial_text` is required.

### Batch Request

Use exactly one of:

1. Legacy prompt list:

```json
{
  "prompts": ["prompt a", "prompt b"],
  "rounds": 3
}
```

2. Structured item list:

```json
{
  "items": [
    {"trial_text": "...", "prediction_target": "success"},
    {"prompt": "..."}
  ],
  "rounds": 3
}
```

### Counterfactual Request

```json
{
  "source_batch_id": "...",
  "source_prompt_id": "...",
  "evidence_ids": ["r1-model-ev-01"],
  "selectors": {
    "models": ["qwen2.5-72b-instruct"],
    "rounds": [1, 2],
    "source_tags": ["primary_endpoint"],
    "include_synthesis": false
  },
  "rounds": 3,
  "allow_fuzzy_quotes": false,
  "metadata": {"experiment": "mask-v1"}
}
```

Selection is union of explicit `evidence_ids` and selector matches.

## Storage Layout

Results are stored at `${DATA_DIR}` (default `/data/results`) with per-batch directories.

Each prompt result now includes:

- `schema_version`
- `request`
- `rounds[]`
- `round_syntheses[]` (chairman synthesis trajectory by round)
- `synthesis`
- `actual_rounds`, `stopped_early`, `early_stop_reason`
- `deliberation_meta`
- `usage_summary` (prompt/completion/total tokens aggregated across rounds + syntheses)
- `evidence_index[]`
- `counterfactual` (when derived from masked rerun)

Per response includes:

- `prediction`
- `evidence[]`
- `structured_json`
- `structured_parse_status`
- `structured_parse_errors`
- `response` (narrative text)
- `usage` (token usage if returned by model endpoint)

`/api/health` and `/api/models` now verify that each endpoint exposes the configured `request_model`.
Readiness only becomes `true` when both endpoint connectivity and request-model resolution pass.

## Verification

```bash
python -c "from backend.config import MODELS, EXTRACTOR_MODEL; print(len(MODELS), EXTRACTOR_MODEL)"
python generate_compose.py
uv run python -m backend.main
curl http://localhost:8001/api/health
```
