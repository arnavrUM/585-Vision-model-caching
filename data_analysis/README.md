# Data Analysis

This directory contains tools and notebooks for analyzing experimental results from the vision model caching project.

## Files

### Notebooks
- **`ablation_results_analysis.ipynb`** - Analyzes ablation study results including latency, cache hit rates, and accuracy across different configurations
- **`model_comparison_analysis.ipynb`** - Compares performance across different vision language models

### Grading Scripts
- **`gqa_llm_regrader.py`** - Re-grades GQA answers using GPT-o3-mini for more nuanced semantic evaluation
  - Processes all CSV logs from experiment directories
  - Grades only unique (question, reference, response) combinations to reduce API costs
  - Outputs graded CSV files with `grader_label` and `match_score` columns

### Result Update Scripts
- **`update_ablation_results.py`** - Updates `ablation_results.csv` with regraded accuracy scores
- **`update_model_comparison_results.py`** - Updates `model_comparison_results.csv` with regraded accuracy scores

## Usage

### Re-grade Ablation Logs
```bash
# Estimate cost first
python gqa_llm_regrader.py \
  --logs-dir ../experiment_logs/ablation_logs \
  --output-dir ../experiment_logs/ablation_logs_graded \
  --estimate-only

# Run grading (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-key-here"
python gqa_llm_regrader.py \
  --logs-dir ../experiment_logs/ablation_logs \
  --output-dir ../experiment_logs/ablation_logs_graded
```

### Update Results with Regraded Accuracy
```bash
python update_ablation_results.py
python update_model_comparison_results.py
```

### Analyze Results
Open the Jupyter notebooks to visualize and analyze the experimental results.

## Rationale

GQA's exact-match evaluation criteria proved overly restrictive, rejecting semantically correct but lexically different responses. We employed GPT-o3-mini as a more flexible grader to better capture answer correctness, resulting in ~10x accuracy improvement (1.7% → 18.2% average accuracy in ablation studies).
