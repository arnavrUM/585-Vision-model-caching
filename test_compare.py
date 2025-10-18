import time
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "./vit_rice_finetuned"
DATASET_NAME = "Subh775/Rice-Disease-Classification-Dataset"
CACHE_THRESHOLD_ALPHA = 0.9  # similarity cutoff
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# LOAD MODEL + FEATURE EXTRACTOR
# ===============================
print("Loading model and feature extractor...")
model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_PATH)

# ===============================
# LOAD DATASET
# ===============================
dataset = load_dataset(DATASET_NAME)
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_dataset = split_dataset["test"]
print(f"Test size: {len(test_dataset)}")

# ===============================
# HELPER: EMBEDDING EXTRACTION
# ===============================
def get_embeddings(images):
    model.eval()
    with torch.no_grad():
        inputs = feature_extractor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model.vit(**inputs)
        # CLS token representation
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        faiss.normalize_L2(embeddings)
    return embeddings

# ===============================
# FAISS CACHE
# ===============================
def create_empty_cache(embedding_dim):
    index = faiss.IndexFlatIP(embedding_dim)
    cached_labels = np.empty((0,), dtype=np.int64)
    return index, cached_labels

def predict_with_cache(image, index, cached_labels, alpha=CACHE_THRESHOLD_ALPHA):
    emb = get_embeddings([image])

    if index.ntotal > 0:
        D, I = index.search(emb, k=1)
        if D[0][0] > alpha:
            pred = int(cached_labels[I[0][0]])
            return pred, True, index, cached_labels

    # Cache miss: do full inference
    inputs = feature_extractor(images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()

    # Add to cache
    index.add(emb)
    cached_labels = np.append(cached_labels, pred)
    return pred, False, index, cached_labels

# ===============================
# BASELINE (NO CACHE)
# ===============================
def predict_baseline(image):
    inputs = feature_extractor(images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    return pred

# ===============================
# RUN COMPARISON
# ===============================
print("Running cache-based inference and baseline comparison...")

# Initialize cache
dummy_emb = get_embeddings([test_dataset[0]["image"]])
index, cached_labels = create_empty_cache(dummy_emb.shape[1])

predictions_cache, ground_truth, cache_hits = [], [], []

# --- Cache-based inference ---
start_cache = time.time()
for sample in tqdm(test_dataset, desc="Cache-based inference"):
    pred, cached, index, cached_labels = predict_with_cache(
        sample["image"], index, cached_labels, alpha=CACHE_THRESHOLD_ALPHA
    )
    predictions_cache.append(pred)
    ground_truth.append(sample["label"])
    cache_hits.append(cached)
end_cache = time.time()
cache_time = end_cache - start_cache

# --- Baseline inference ---
start_baseline = time.time()
predictions_baseline = []
for sample in tqdm(test_dataset, desc="Baseline inference"):
    pred = predict_baseline(sample["image"])
    predictions_baseline.append(pred)
end_baseline = time.time()
baseline_time = end_baseline - start_baseline

# ===============================
# METRICS + REPORT
# ===============================
acc_cache = accuracy_score(ground_truth, predictions_cache)
acc_baseline = accuracy_score(ground_truth, predictions_baseline)
cache_hit_rate = np.mean(cache_hits)

print("\n=== PERFORMANCE COMPARISON ===")
print(f"Cache-based Accuracy:  {acc_cache:.4f}")
print(f"Baseline Accuracy:     {acc_baseline:.4f}")
print(f"Cache Hit Rate:        {cache_hit_rate:.4f}")
print(f"⏱️ Cache Inference Time: {cache_time:.2f} s")
print(f"⏱️ Baseline Time:        {baseline_time:.2f} s")
print(f"🚀 Speedup: {baseline_time / cache_time:.2f}× faster")

# Save results
results = pd.DataFrame({
    "true_label": ground_truth,
    "cache_pred": predictions_cache,
    "baseline_pred": predictions_baseline,
    "cached": cache_hits
})
results.to_csv("cache_vs_baseline_results.csv", index=False)
print("Results saved to cache_vs_baseline_results.csv")
