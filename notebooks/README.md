# Analysis Notebooks

This directory contains Jupyter notebooks for analyzing and visualizing Multi-Dimensional AI training data and results.

## Available Notebooks

### training_analysis.ipynb

Analyze training metrics and model performance:

- Load and visualize loss curves per modality
- Analyze throughput (samples/sec) over training
- Inspect checkpoint states
- Compare different training runs

**Usage:**

```bash
jupyter notebook notebooks/training_analysis.ipynb
```

### dataset_exploration.ipynb

Explore and visualize dataset samples:

- Load synthetic or real multi-modal data
- Visualize vision, audio, and sensor inputs
- Analyze data distributions
- Validate data quality and ranges

**Usage:**

```bash
jupyter notebook notebooks/dataset_exploration.ipynb
```

## Setup

Install Jupyter if not already available:

```bash
pip install jupyter matplotlib pandas
```

Launch Jupyter from project root:

```bash
cd "Multi Dimensional AI"
jupyter notebook
```

## Creating New Notebooks

When creating analysis notebooks:

1. Add `sys.path.append('..')` to import project modules
2. Use relative paths from project root
3. Document the notebook's purpose in markdown cells
4. Include example visualizations
