pip install torch torchvision transformers datasets scikit-learn pillow tqdm


# Experiment 2
Setup
~~~
# Install uv if not availale
uv venv --seed
source .venv/bin/activate
uv pip install -r requirements.txt
~~~

## Ablation Study
36 conditions
- qwen
    - exact text match
    - fusion
    - semantic (threshold: 0.5)
    - semantic (threshold: 0.6)
    - semantic (threshold: 0.7)
    - semantic (threshold: 0.9)
    - embedding (prompt threshold: 0.7, vision threshold: 0.7)
    - embedding (prompt threshold: 0.7, vision threshold: 0.8)
    - embedding (prompt threshold: 0.7, vision threshold: 0.9)
    - embedding (prompt threshold: 0.8, vision threshold: 0.7)
    - embedding (prompt threshold: 0.8, vision threshold: 0.8)
    - embedding (prompt threshold: 0.8, vision threshold: 0.9)
    - embedding (prompt threshold: 0.9, vision threshold: 0.7)
    - embedding (prompt threshold: 0.9, vision threshold: 0.8)
    - embedding (prompt threshold: 0.9, vision threshold: 0.9)
- internvl
    - ...
~~~
chmod +x ./experiment2/ablation.sh
./experiment2/ablation.sh
~~~

## Model Comparison Study
6 conditions (with all techniques on)
- qwen-2B
- qwen-4B
- qwen-8B
- internvl-2B
- internvl-4B
- internvl-8B
~~~
chmod +x ./experiment2/model_comparison.sh
./experiment2/model_comparison.sh
~~~