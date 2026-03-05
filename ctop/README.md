Using the Council and Counterfactual pattern to predict clinical trials.

## CTOP XML Preprocessing

Build the intermediate trial dataframe from ClinicalTrials.gov XML files:

```bash
uv run python -m ctop.build_trials_dataframe \
  --input-glob "data/trials/TOP/*.xml" \
  --output-dir "/data/trials/output" \
  --strict
```

Outputs:

- `/data/trials/output/trials_intermediate.parquet`
- `/data/trials/output/trials_intermediate.csv`

## CTOP Duration LLM Pipeline

Run duration prediction with automated per-evidence counterfactuals enabled by default:

```bash
uv run python -m ctop.run_duration_pipeline \
  --input-path "/data/trials/output/trials_intermediate.parquet" \
  --output-root "/data/trials/output"
```

Default runtime UI uses a two-line progress display:

- line 1: tqdm progress bar with live metrics (`ema@1m`, `hit@1m`, `mae`, `rmse`, `f1@12m`)
- line 2: latest baseline response summary

Tune metric thresholds with:

- `--epsilon-months` (default `1.0`)
- `--ema-alpha` (default `0.12`)
- `--short-threshold-months` (default `12.0`)

Each run writes to a UUID-named directory:

- `/data/trials/output/{RUN_ID}/duration_predictions_long.csv`
- `/data/trials/output/{RUN_ID}/duration_predictions_long.parquet`
- `/data/trials/output/{RUN_ID}/duration_predictions_long.pkl`
- `/data/trials/output/{RUN_ID}/duration_predictions_wide.csv`
- `/data/trials/output/{RUN_ID}/duration_predictions_wide.parquet`
- `/data/trials/output/{RUN_ID}/duration_predictions_wide.pkl`
- `/data/trials/output/{RUN_ID}/run_manifest.json`
