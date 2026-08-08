# ML

My personal machine learning sandbox.

This repository is where I experiment with machine learning concepts, implement things I'm learning, explore ideas, and build personal projects.

It is intentionally a mix of experiments, exploratory work, and larger projects rather than a polished ML library.

## Structure

```text
ML/
├── experiments/     # Small, focused ML experiments
├── notebooks/       # Exploratory notebooks and visualizations
├── projects/        # Larger personal ML projects
├── src/             # Reusable ML code
├── datasets/        # Local datasets (not tracked)
├── models/          # Generated model artifacts (not tracked)
├── README.md
└── requirements.txt
```

## Experiments

Experiments are numbered chronologically and focus on a specific concept or question.

Examples include:

* Neural networks
* Backpropagation
* Optimization
* Classical machine learning
* Deep learning
* Model evaluation
* Implementations from papers and other learning material

## Projects

Larger pieces of work that grow beyond a single experiment live here.

## Environment

The repository uses Python and common scientific/ML libraries including NumPy, pandas, scikit-learn, SciPy, Matplotlib, Jupyter, and PyTorch.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Create and activate a virtual environment before installing dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

