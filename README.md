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