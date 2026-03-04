# LLM Council (Local NIM/vLLM)

![llmcouncil](header.jpg)

Backend-only refactor of LLM Council that runs local NVIDIA NIM containers (vLLM backend) and orchestrates configurable multi-round deliberation.

## What Changed

- Replaced OpenRouter cloud inference with local OpenAI-compatible NIM endpoints.
- Replaced fixed 3-stage flow with configurable `N` rounds from `models.yaml`.
- Added batch deliberation endpoints and layered JSON storage under `/data/results`.
- Removed frontend dependency from startup flow.

## Configuration

### 1. Create environment file

```bash
cp .env.example .env
```

Set:

- `NGC_API_KEY` (required for NIM image/model pulls)
- `DATA_DIR` (optional, defaults to `/data/results`)
- `MODELS_YAML` (optional, defaults to `./models.yaml`)

### 2. Edit `models.yaml`

`models.yaml` is the single source of truth for:

- round count (`rounds`)
- model names
- request model ids (`request_model`, optional)
- NIM images
- GPU assignments
- ports
- chairman selection

## Generate Compose

```bash
python generate_compose.py
```

This writes `docker-compose.yml` from `models.yaml`.

## Run

```bash
uv sync
./start.sh
```

`start.sh`:

1. runs `docker compose up -d`
2. starts the FastAPI backend on `http://localhost:8001`
3. waits until `/api/health` reports all model endpoints ready

## API Endpoints

- `POST /api/deliberate` - run one prompt through N rounds
- `POST /api/batch` - run a list of prompts sequentially
- `GET /api/batch/{batch_id}` - fetch stored batch results
- `GET /api/models` - model config + live health checks
- `GET /api/health` - overall readiness

## Storage Layout

Results are stored at `${DATA_DIR}` (default `/data/results`):

- one directory per batch id
- `batch.json` manifest
- one JSON file per prompt containing:
  - `rounds[]`
  - `synthesis`
  - metadata (`batch_id`, `prompt_id`, timestamps)

## Verification

```bash
python -c "from backend.config import MODELS; print(MODELS)"
python generate_compose.py
uv run python -m backend.main
```

Then test endpoints with `curl`:

```bash
curl http://localhost:8001/api/health
```
