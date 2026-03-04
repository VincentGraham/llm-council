# Plan: Local NIM/vLLM Multi-Round Council Refactor

Replace OpenRouter cloud API with local NIM containers (vLLM backend) on 8×A100. Refactor fixed 3-stage council into configurable N-round deliberation with batch prompt support. Single `models.yaml` config. Backend only — no frontend.

---

## Phase 1: Infrastructure — Docker + Model Config

**Step 1** — Create `models.yaml` (project root). Single source of truth: model names, NIM images, GPU assignments, ports, round count, chairman designation. Users edit only this file to swap models.

**Step 2** — Create `generate_compose.py`. Reads `models.yaml`, outputs `docker-compose.yml`. Each model becomes a NIM service following your provided pattern (`runtime: nvidia`, `ipc: host`, ulimits, cache env vars pointing at `/opt/nim/.cache`, volume `/data/nim-cache:/opt/nim/.cache`, `NGC_API_KEY`, `NIM_INFERENCE_BACKEND=vllm`). Port mapping: `{model.port}:8000`. GPU assignment via `NVIDIA_VISIBLE_DEVICES`.

**Step 3** — Generate initial `docker-compose.yml` from default models.yaml.

## Phase 2: Backend Inference Client

**Step 4** — Replace `backend/openrouter.py` → `backend/inference.py`. Remove OpenRouter auth. `query_model()` now hits `http://localhost:{port}/v1/chat/completions` (NIM's OpenAI-compatible API). No auth needed for local containers. Add `health_check()` per model. Same return shape for downstream compatibility.

**Step 5** — Replace `backend/config.py`. Remove `OPENROUTER_API_KEY`, hardcoded model lists. Load `models.yaml` via PyYAML. Expose `MODELS`, `COUNCIL_MODELS`, `CHAIRMAN_MODEL`, `ROUNDS`, `MODEL_ENDPOINTS` dict mapping name → local URL.

**Step 6** — Add `pyyaml>=6.0` to `pyproject.toml`.

## Phase 3: Multi-Round Deliberation Engine

**Step 7** — Refactor `backend/council.py`. Replace fixed stage1/stage2/stage3 with:
- `round_1(prompt)` — all models answer independently in parallel (same as old stage1)
- `round_n(prompt, prior_rounds, n)` — each model sees original prompt + anonymized responses from OTHER models from all prior rounds; evaluates/critiques/refines
- `synthesize(prompt, all_rounds)` — chairman sees everything, produces final answer
- `run_deliberation(prompt, rounds=N)` — orchestrates: round 1 → store → round 2 → store → ... → synthesis → store
- `run_batch_deliberation(prompts, rounds=N)` — sequential per-prompt (GPU memory shared); returns all results

Keep `parse_ranking_from_text()` and `calculate_aggregate_rankings()` for cross-review rounds. Remove `generate_conversation_title()`.

**Step 8** — Refactor `backend/storage.py`. Store on `/data/results/` (raid array). New schema: per-prompt JSON with `rounds[]` array (each round = a stored layer with all model responses + metadata) and `synthesis` object. Functions: `save_round()`, `save_synthesis()`, `load_result()`, `list_batches()`.

## Phase 4: API Endpoints

**Step 9** — Refactor `backend/main.py`. New endpoints:
- **`POST /api/deliberate`** — single prompt, returns full N-round result
- **`POST /api/batch`** — list of prompts, returns batch results
- **`GET /api/batch/{batch_id}`** — retrieve stored batch
- **`GET /api/models`** — list models with health status
- **`GET /api/health`** — overall system readiness

Remove CORS (no frontend). Remove old conversation/streaming endpoints.

## Phase 5: Cleanup

**Step 10** — Update `start.sh`: remove frontend, add `docker compose up -d` + health wait loop. Update root `main.py` as optional CLI entrypoint.

**Step 11** — Create `.env.example` documenting `NGC_API_KEY` and optional `DATA_DIR`. Update README.

---

## Relevant Files

| File | Action |
|---|---|
| `models.yaml` | **Create** — model config |
| `generate_compose.py` | **Create** — compose generator |
| `docker-compose.yml` | **Create** (generated) |
| `backend/inference.py` | **Create** (replaces openrouter.py) |
| `backend/config.py` | **Rewrite** — YAML loader |
| `backend/council.py` | **Rewrite** — N-round engine |
| `backend/storage.py` | **Rewrite** — layered storage |
| `backend/main.py` | **Rewrite** — new API endpoints |
| `pyproject.toml` | **Edit** — add pyyaml |
| `start.sh` | **Edit** — docker compose + backend only |
| `.env.example` | **Create** |

## Verification

1. `python -c "from backend.config import MODELS; print(MODELS)"` — YAML parses
2. `python generate_compose.py && docker compose config` — valid compose
3. `docker compose up -d && docker compose ps` — containers healthy
4. `curl http://localhost:{port}/v1/models` per container — NIM responds
5. `POST /api/deliberate` with test prompt — full N-round result
6. `POST /api/batch` with 2-3 prompts — results stored in `/data/results/`
7. Inspect JSON in `/data/results/` — round layers + synthesis present
8. `nvidia-smi` during inference — models on assigned GPUs

## Further Considerations

1. **Async batch processing** — current plan blocks the HTTP response. For very large batches, could add a job queue + status polling later. Starting synchronous is simpler.
2. **Model warm-up** — NIM containers take minutes to load weights. `start.sh` should have a `--wait` flag that polls `/v1/models` until all containers report ready.
3. **GPU over-subscription** — models.yaml lets users assign overlapping GPU IDs. Could add a validation step in `generate_compose.py` that warns on overlap.
