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
