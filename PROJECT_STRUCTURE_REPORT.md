# EnergyAwareUPF Project Structure Report

Generated: 2026-03-23

## 1) Executive Summary

EnergyAwareUPF is an RL-centered research-and-engineering codebase for energy-aware UPF selection in 5G systems. The repository is organized around four major layers:

- Core product code (`src/`): environment, agent, training, evaluation.
- Operational scripts (`scripts/`): experiment orchestration, batch runs, baseline pipelines, and analysis helpers.
- Data and model assets (`data/`, `saved_models/`, `finalModels/`, `traffic_predictor/`): input datasets, trained model artifacts, and traffic forecasting assets.
- Research outputs (`experiments/results/`, `polished_results/`, `Paper artifacts/`, notebooks): reproducible experimental outputs and publication material.

This structure supports both reproducible research and production-like experimentation workflows.

## 2) Top-Level Tree (Condensed)

```text
EnergyAwareUPF/
├── src/                        # RL core modules
├── scripts/                    # Runbooks, orchestration, analysis scripts
├── tests/                      # Unit/integration tests
├── docs/                       # Technical documentation
├── data/
│   ├── raw/                    # Source traffic/performance data
│   └── processed/              # Preprocessed datasets
├── experiments/
│   ├── sweep_config*.yaml      # Hyperparameter and scenario sweeps
│   └── results/                # Main experiment artifacts (large)
├── saved_models/               # Model checkpoints and predictor assets
├── finalModels/                # Final/selected model snapshots
├── traffic_predictor/          # Traffic prediction pipeline + outputs
├── notebooks/                  # Analysis and paper-results notebooks
├── polished_results/           # Curated figures/tables for reporting
├── results/                    # Additional summarized result outputs
├── Paper artifacts/            # Manuscript, references, publication figures
├── config.yaml                 # Main runtime/training configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview and usage
```

## 3) Folder Responsibilities

### Core Runtime and Learning

- `src/agent.py`: PPO/RecurrentPPO model creation and loading.
- `src/environment.py`: `ManualCooldownEnv` with action cooldown and SEC-oriented reward.
- `src/utils.py`: data loading/preprocessing, feature banks, vectorized env setup.
- `src/train.py`: full training entry pipeline.
- `src/evaluate.py`: policy rollout, metrics, CSV/figure artifact generation.
- `src/offline_optimal.py`: dynamic-programming baseline under same env semantics.

### Automation and Workflows

- `scripts/run_experiments.py`: executes configured experiment suites.
- `scripts/run_batch_evaluation.py`: evaluates multiple runs in batch mode.
- `scripts/run_offline_optimal_baseline.py`: computes exact offline-optimal baseline.
- `scripts/baseline_rule.py`: rule-based baseline strategy.
- `scripts/analyze_experiments.py`: cross-run aggregation/analysis.

### Validation and Quality

- `tests/`: unit and integration tests for environment, agent, evaluation, and offline baseline logic.
- `TEST_RESULTS_SUMMARY.txt`: test run summary artifact.

### Documentation

- `docs/API_REFERENCE.md`: API signatures and behavior references.
- `docs/MODULES.md`: module-level deep dives.
- `docs/DATA_FLOW.md`: end-to-end data and control flow.
- `docs/DEPENDENCIES.md`: dependency mapping.

### Data, Models, and Forecasting

- `data/raw/`: source datasets.
- `data/processed/`: transformed datasets used by training/evaluation.
- `saved_models/prediction_models/`: learned prediction model assets.
- `finalModels/`: final selected model outputs.
- `traffic_predictor/`: independent predictor training/inference artifacts and configs.

### Results and Publication Assets

- `experiments/results/`: primary artifact store for sweep and run outputs.
- `polished_results/`: curated assets for final reporting.
- `Paper artifacts/`: manuscript source and publication-grade figures.
- `notebooks/`: exploratory and paper assembly analytics.

## 4) Ownership-by-Folder Matrix

| Area | Primary Owner Role | Typical Activities |
|---|---|---|
| `src/` | RL/ML engineer | Environment design, reward shaping, training/eval logic |
| `scripts/` | MLOps/research engineer | Running sweeps, baselines, batch analysis |
| `tests/` | QA + contributors | Unit/integration regression coverage |
| `docs/` | Maintainers/authors | API and architecture documentation |
| `data/` | Data engineer/research engineer | Data ingestion and preprocessing |
| `saved_models/`, `finalModels/` | ML engineer | Model lifecycle, checkpointing, final export |
| `experiments/results/` | Research team | Raw experiment trace and metrics archive |
| `polished_results/`, `Paper artifacts/` | Research authors | Final figures, manuscript, publication outputs |
| `traffic_predictor/` | Forecasting owner | Traffic model training, evaluation, metadata tracking |

## 5) Repository Scale Snapshot

- Top-level files: 19
- File count by major directory:
  - `experiments/`: 1188
  - `polished_results/`: 104
  - `finalModels/`: 32
  - `scripts/`: 29
  - `Paper artifacts/`: 27
  - `traffic_predictor/`: 24
  - `src/`: 18
  - `saved_models/`: 18
  - `tests/`: 17
  - `docs/`: 5
  - `data/`: 5
  - `notebooks/`: 5
  - `results/`: 4
  - `logs/`: 1

## 6) Execution Path (Conceptual)

1. Configure run parameters in `config.yaml` and optional sweep files.
2. Preprocess data and build environment inputs through `src/utils.py`.
3. Train policy through `src/train.py` and `src/agent.py`.
4. Evaluate policy and produce artifacts through `src/evaluate.py`.
5. Compare against rule-based and offline-optimal baselines via scripts.
6. Consolidate outputs into `experiments/results/`, then curate in `polished_results/` and paper assets.

## 7) Suggested Conventions (Optional)

- Keep all new runnable entry points in `scripts/` and pure library logic in `src/`.
- Store large generated files only in results/artifact folders, not in `src/` or `tests/`.
- Add any new architecture decisions to `docs/` to preserve reproducibility.
- For every new pipeline script, add at least one integration test or smoke test in `tests/`.
