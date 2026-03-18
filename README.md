# EnergyAwareUPF

A reinforcement learning framework for intelligent User Plane Function (UPF) selection in 5G networks, optimizing the trade-off between energy efficiency and Quality-of-Service.

## Overview

EnergyAwareUPF uses Proximal Policy Optimization (PPO) to learn adaptive policies that dynamically select between different UPF implementations (DPDK and OAI) based on real-time traffic patterns, capacity constraints, and temporal features. The system maximizes Specific Energy Consumption (SEC) - throughput per watt - while enforcing QoS guarantees.

**Key Features:**
- 🎯 Energy-aware decision making with SEC-based rewards
- 🔒 QoS threshold enforcement with penalty mechanisms
- 📊 Traffic-aware adaptation using historical patterns
- ⏰ Temporal intelligence via calendar feature encoding
- 🚀 Fast training with vectorized environments and precomputed lookups
- 📈 Production-ready with comprehensive evaluation tools

**Use Cases:**
- 5G network energy optimization
- Dynamic workload management
- Green networking research
- Intelligent resource allocation

## 📊 Documentation

- **[Comprehensive Report](COMPREHENSIVE_REPORT.md)** - Detailed analysis of test results, configuration options, and best practices ⭐
- [API Reference](docs/API_REFERENCE.md) - Function signatures and documentation
- [Modules Guide](docs/MODULES.md) - Deep dive into each component
- [Testing Guide](tests/README.md) - How to run and write tests

## Offline-Optimal Baseline

You can compute an exact offline-optimal controller (dynamic programming under the same `ManualCooldownEnv` cooldown/reward semantics) and compare it against PPO and rule-based baselines:

```bash
python -m scripts.run_offline_optimal_baseline \
	--run-dir experiments/results/D1_mlp_lstm_cd4_k_threshold/mlp_hybrid_cd4_k2_t90/seed_456 \
	--tag d1_mlp_hybrid_cd4_k2_t90
```

The script writes artifacts under `experiments/results/Baselines/`, including:

- `offline_optimal_eval_<tag>.csv` (full per-step replay under env semantics)
- `baseline_offline_optimal_results_<tag>.csv` (baseline-style timeline CSV)
- `baseline_offline_optimal_summary_<tag>.csv` (summary metrics)
- `offline_optimal_vs_baselines_<tag>.csv` (offline vs PPO vs rule comparison)
- `offline_optimal_sanity_checks_<tag>.json` (sanity-check outcomes)