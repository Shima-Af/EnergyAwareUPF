# Testing Guide — EnergyAwareUPF

This document covers **how to run the project tests**, the **fundamentals of pytest**, and **why testing matters in ML engineering**.

---

## Table of Contents

1. [Running the Tests](#1-running-the-tests)
2. [Test Suite Overview](#2-test-suite-overview)
3. [Pytest Fundamentals (Study Material)](#3-pytest-fundamentals-study-material)
4. [Testing in ML Engineering (Study Material)](#4-testing-in-ml-engineering-study-material)
5. [Recommended Resources](#5-recommended-resources)

---

## 1. Running the Tests

### Prerequisites

Make sure you have the virtual environment activated and dependencies installed:

```bash
source .venv/bin/activate
pip install pytest pytest-cov pytest-mock
```

### Common Commands

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report (terminal + HTML)
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run a specific test file
pytest tests/test_environment.py -v

# Run a specific test class
pytest tests/test_environment.py::TestManualCooldownEnv -v

# Run a single test method
pytest tests/test_environment.py::TestManualCooldownEnv::test_env_step -v

# Run only fast tests (skip slow integration tests)
pytest tests/ -v -m "not slow"

# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Stop on first failure
pytest tests/ -v -x

# Show print statements during tests
pytest tests/ -v -s

# Run with short traceback (easier to read)
pytest tests/ -v --tb=short
```

---

## 2. Test Suite Overview

| File | What It Tests | Type |
|------|--------------|------|
| `conftest.py` | Shared fixtures: config, traffic data, lookup tables | Fixtures |
| `test_utils.py` | Config loading, traffic dynamics, calendar features | Unit |
| `test_environment.py` | Env init, reset, step, cooldown, QoS penalties | Unit |
| `test_evaluation.py` | DPDK/OAI baseline prediction functions | Unit |
| `test_agent.py` | PPO/RecurrentPPO agent creation, save/load | Unit |
| `test_integration.py` | Full training pipeline end-to-end | Integration |

### Project Test Architecture

```
tests/
├── conftest.py              ← Shared fixtures (auto-discovered by pytest)
├── test_utils.py            ← Pure function tests (fastest)
├── test_evaluation.py       ← Prediction/lookup tests
├── test_environment.py      ← RL environment lifecycle tests
├── test_agent.py            ← Agent creation/persistence tests
└── test_integration.py      ← End-to-end pipeline tests (slowest)
```

---

## 3. Pytest Fundamentals (Study Material)

### 3.1 What Is Pytest?

Pytest is Python's most popular testing framework. It discovers and runs functions/methods that start with `test_`, and uses Python's built-in `assert` statement to verify behavior.

### 3.2 Test Structure — Arrange, Act, Assert

Every well-written test follows three phases:

```python
def test_example():
    # ARRANGE — set up inputs and expected outputs
    traffic = np.array([10.0, 20.0, 30.0])
    
    # ACT — call the function being tested
    result = compute_traffic_dynamics(traffic, window_size=2)
    
    # ASSERT — verify the output matches expectations
    assert 'feature_names' in result
    assert len(result['dT']) == len(traffic)
```

### 3.3 Fixtures

Fixtures are reusable setup functions that provide test data. Pytest injects them automatically by matching parameter names:

```python
# In conftest.py — available to ALL test files automatically
@pytest.fixture
def sample_config():
    return {'training': {'seed': 42}, ...}

# In test file — pytest injects `sample_config` automatically
def test_load_config(sample_config):
    assert 'training' in sample_config
```

**Key fixture concepts:**
- **`conftest.py`** — fixtures defined here are auto-discovered by all tests in the same directory
- **`tmp_path`** — built-in pytest fixture that provides a temporary directory (auto-cleaned)
- **Scope** — fixtures can be scoped to `function` (default), `class`, `module`, or `session`

```python
@pytest.fixture(scope="session")  # Created once for entire test run
def expensive_model():
    return load_heavy_model()
```

### 3.4 Parametrize — Testing Multiple Inputs

Instead of writing duplicate tests, use `@pytest.mark.parametrize`:

```python
@pytest.mark.parametrize("throughput", [0.0, 10.0, 50.0, 100.0])
def test_prediction_multiple_inputs(throughput):
    perf, power = predict_dpdk_from_lookup(lookup, throughput)
    assert perf >= 0
    assert power >= 0
```

This generates 4 separate test cases from one function.

### 3.5 Markers — Categorizing Tests

Markers let you tag and selectively run tests:

```python
@pytest.mark.slow          # Custom marker for slow tests
def test_full_training():
    ...

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_gpu_training():
    ...
```

Run selectively:
```bash
pytest -m "not slow"       # Skip slow tests
pytest -m "slow"           # Run only slow tests
```

### 3.6 Key Assertions

```python
# Value checks
assert result == expected
assert result > 0
assert result is not None

# Type checks
assert isinstance(result, float)

# Container checks
assert 'key' in dictionary
assert len(my_list) == 5

# Approximate float comparison
assert result == pytest.approx(3.14, abs=1e-2)

# Exception checks
with pytest.raises(ValueError):
    function_that_should_fail()

# NumPy array comparison
np.testing.assert_array_almost_equal(arr1, arr2, decimal=5)
```

### 3.7 conftest.py — The Fixture Hub

`conftest.py` is a **special file** that pytest auto-discovers:

- Fixtures defined here are available to **all test files** in the same directory (no import needed)
- You can have multiple `conftest.py` files at different directory levels
- It's the right place for shared setup, mock data, and test configuration

---

## 4. Testing in ML Engineering (Study Material)

### 4.1 Why Testing Matters in ML

ML systems are harder to test than traditional software because:
- **Non-determinism** — random seeds, GPU float precision, stochastic training
- **Data dependency** — model behavior depends on training data quality
- **Hidden feedback loops** — a model's output can affect its future input
- **Slow iteration** — training takes hours/days; you can't manually verify each change

Testing is your **safety net** — it catches regressions, validates assumptions, and makes refactoring possible.

### 4.2 The ML Testing Pyramid

```
        ╱ ╲
       ╱   ╲         End-to-End Tests
      ╱ E2E ╲        (full training pipeline, slow)
     ╱───────╲
    ╱         ╲       Integration Tests
   ╱Integration╲     (components working together)
  ╱─────────────╲
 ╱               ╲    Unit Tests
╱   Unit Tests    ╲   (individual functions, fast)
╲─────────────────╱
```

- **Unit tests** (most tests) — test individual functions in isolation. Fast, cheap, run often.
- **Integration tests** — test that components work together (env + agent, data pipeline + model).
- **End-to-end tests** (fewest tests) — test the full pipeline from data loading to evaluation.

### 4.3 What to Test in an ML Project

#### A. Data Tests
Validate data at every stage of your pipeline:

```python
def test_no_missing_values():
    df = load_data("traffic.csv")
    assert df.isnull().sum().sum() == 0

def test_traffic_values_in_range():
    traffic = load_traffic()
    assert traffic.min() >= 0
    assert traffic.max() <= 10000  # physically reasonable
    
def test_data_shape():
    X_train, y_train = load_split()
    assert X_train.shape[0] == y_train.shape[0]
```

#### B. Model Tests
Verify model behavior without full training:

```python
def test_model_output_shape():
    model = create_model()
    dummy_input = np.zeros((1, 10))
    output = model.predict(dummy_input)
    assert output.shape == (1, 1)

def test_model_predicts_positive_power():
    """Power consumption can't be negative."""
    output = model.predict(sample_input)
    assert output >= 0

def test_loss_decreases():
    """Verify that model can learn (overfit on small batch)."""
    model = create_model()
    loss_before = evaluate(model)
    model.fit(small_batch, epochs=10)
    loss_after = evaluate(model)
    assert loss_after < loss_before
```

#### C. Environment Tests (RL-specific)
This is what your `test_environment.py` does — critical for RL:

```python
def test_observation_space():
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

def test_reward_is_finite():
    _, reward, _, _, _ = env.step(action)
    assert np.isfinite(reward)

def test_episode_terminates():
    """Ensure the episode ends eventually."""
    done = False
    steps = 0
    env.reset()
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert steps > 0
```

#### D. Reproducibility Tests
Ensure deterministic results with fixed seeds:

```python
def test_deterministic_predictions():
    result1 = predict_with_seed(42)
    result2 = predict_with_seed(42)
    np.testing.assert_array_equal(result1, result2)
```

#### E. Performance / Regression Tests
Guard against performance degradation:

```python
def test_inference_latency():
    import time
    start = time.time()
    model.predict(batch)
    elapsed = time.time() - start
    assert elapsed < 0.1  # 100ms budget

def test_model_accuracy_above_threshold():
    accuracy = evaluate_model(model, test_data)
    assert accuracy > 0.85  # minimum acceptable
```

### 4.4 ML-Specific Testing Challenges

| Challenge | Solution |
|-----------|----------|
| **Floating-point precision** | Use `pytest.approx()` or `np.testing.assert_allclose()` |
| **Randomness** | Fix seeds in tests; test statistical properties not exact values |
| **Slow training** | Use tiny datasets and few epochs for tests; mark slow tests |
| **Large models** | Mock heavy models in unit tests; test with small architectures |
| **GPU dependency** | Use `@pytest.mark.skipif` for GPU-only tests |
| **External data** | Use fixtures with synthetic data (like your `sample_traffic_data`) |

### 4.5 Testing Anti-Patterns to Avoid

- **Testing implementation, not behavior** — Don't assert on internal variable values; assert on outputs
- **Flaky tests** — Tests that sometimes pass, sometimes fail due to randomness
- **Testing too much at once** — Each test should verify ONE thing
- **No assertions** — A test that runs code but never asserts is useless
- **Hardcoded file paths** — Use `tmp_path` fixture for temporary files

---

## 5. Recommended Resources

### Pytest
- [Pytest Official Documentation](https://docs.pytest.org/en/stable/) — the authoritative reference
- [Pytest fixtures guide](https://docs.pytest.org/en/stable/how-to/fixtures.html) — deep dive into fixtures
- [Real Python: Testing with Pytest](https://realpython.com/pytest-python-testing/) — beginner-friendly tutorial
- [Effective Python Testing with Pytest](https://realpython.com/pytest-python-testing/) — comprehensive walkthrough

### ML Testing
- [Google: Testing ML Systems (NIPS 2016)](https://research.google/pubs/pub45742/) — foundational paper on ML testing
- [Made With ML: Testing](https://madewithml.com/courses/mlops/testing/) — practical ML testing patterns
- [Evidently AI: ML Model Testing Guide](https://www.evidentlyai.com/ml-system-design/ml-model-testing) — model validation strategies
- [Jeremy Jordan: Testing ML Systems](https://www.jeremyjordan.me/testing-ml/) — excellent blog post on the topic
- [Microsoft: ML Testing Best Practices](https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines) — enterprise perspective

### Books
- *Python Testing with pytest* by Brian Okken — the definitive pytest book
- *Reliable Machine Learning* by Cathy Chen et al. (O'Reilly) — production ML including testing
- *Designing Machine Learning Systems* by Chip Huyen (O'Reilly) — Chapter 9 covers testing and monitoring