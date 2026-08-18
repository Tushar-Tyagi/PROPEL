# PROPEL Development Guide

## Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tushar-Tyagi/PROPEL.git
   cd PROPEL
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install in editable mode with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

Run the test suite with pytest:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest --cov=propel tests/
```

## Running Examples

- **Quickstart:**
  ```bash
  python examples/quickstart.py
  ```

- **Offline Probing:**
  ```bash
  python examples/offline_probing.py
  ```

- **Dataset Evaluation:**
  ```bash
  python examples/evaluate_dataset.py
  ```
