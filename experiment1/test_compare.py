import time
import re
import torch
import torchvision
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("[WARN] sentence-transformers not available. L1 semantic cache will be disabled.")

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "./vit_rice_finetuned"
DATASET_NAME = "Subh775/Rice-Disease-Classification-Dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Encoders to test
ENCODERS = ["resnet18", "mobilenet_v2", "efficientnet_b0"]

# Threshold values to test
THRESHOLDS = [0.875, 0.9, 0.925, 0.95]

# ===============================
# LOAD MODEL + FEATURE EXTRACTOR
# ===============================
print("Loading model and feature extractor...")
model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_PATH)

# ===============================
# LOAD LIGHT ENCODER FOR EMBEDDINGS
# ===============================
def load_encoder(encoder_name):
    """Load and configure lightweight encoder for embeddings"""
    print(f"Loading {encoder_name} encoder for embeddings...")
    
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    try:
        if encoder_name == "resnet18":
            try:
                weights = torchvision.models.ResNet18_Weights.DEFAULT
                encoder_model = torchvision.models.resnet18(weights=weights)
            except (AttributeError, TypeError):
                encoder_model = torchvision.models.resnet18(pretrained=True)
            encoder_model = torch.nn.Sequential(*list(encoder_model.children())[:-1])
            img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
            
        elif encoder_name == "mobilenet_v2":
            try:
                weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
                encoder_model = torchvision.models.mobilenet_v2(weights=weights)
            except (AttributeError, TypeError):
                encoder_model = torchvision.models.mobilenet_v2(pretrained=True)
            # MobileNetV2 structure is different - use features module directly
            encoder_model = encoder_model.features
            # Add global average pooling
            encoder_model = torch.nn.Sequential(
                encoder_model,
                torch.nn.AdaptiveAvgPool2d((1, 1))
            )
            img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
            
        elif encoder_name == "efficientnet_b0":
            try:
                weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
                encoder_model = torchvision.models.efficientnet_b0(weights=weights)
            except (AttributeError, TypeError):
                encoder_model = torchvision.models.efficientnet_b0(pretrained=True)
            # EfficientNet structure: features -> avgpool -> classifier
            # We want features + avgpool
            encoder_model = torch.nn.Sequential(
                encoder_model.features,
                encoder_model.avgpool
            )
            img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}. Choose from: 'resnet18', 'mobilenet_v2', 'efficientnet_b0'")
        
        encoder_model = encoder_model.to(DEVICE)
        encoder_model.eval()
        return encoder_model, img_transform
        
    except Exception as e:
        raise RuntimeError(f"Failed to load encoder {encoder_name}: {e}")

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
def get_embeddings(images, encoder, encoder_transform):
    """Extract embeddings using the selected lightweight encoder"""
    encoder.eval()
    with torch.no_grad():
        # Convert PIL images to tensors and apply encoder-specific preprocessing
        if isinstance(images, list):
            processed_images = torch.stack([encoder_transform(img.convert('RGB')) for img in images])
        else:
            processed_images = encoder_transform(images.convert('RGB')).unsqueeze(0)
        
        processed_images = processed_images.to(DEVICE)
        embeddings = encoder(processed_images)
        # Flatten the spatial dimensions (handles different output shapes)
        embeddings = embeddings.view(embeddings.size(0), -1).detach().cpu().numpy()
        faiss.normalize_L2(embeddings)
    return embeddings

# ===============================
# HIERARCHICAL CACHE: L0.5 (Exact), L1 (Semantic), L2 (Embedding)
# ===============================
# Cache is in-memory only, no disk persistence
# Each trial starts with an empty cache
# Frame N can only access cached results from frames 0 to N-1 in the same trial
# Lookup order: L0.5 exact → L1 semantic → L2 embedding
# Vision embedding (L2) is always checked even if text cache hits

class ExactTextCache:
    """L0.5: Exact text cache for normalized text matching."""
    def __init__(self):
        self._space_re = re.compile(r"\s+")
        self._index = {}  # normalized_text -> list of (chunk_id, prediction)
    
    def normalize(self, text: str | None) -> str:
        """Normalize text for exact matching."""
        normalized = (text or "").strip().lower()
        if not normalized:
            return ""
        return self._space_re.sub(" ", normalized)
    
    def add(self, normalized: str, chunk_id: str, prediction: int):
        """Add entry to exact cache."""
        if not normalized:
            return
        if normalized not in self._index:
            self._index[normalized] = []
        self._index[normalized].append((chunk_id, prediction))
    
    def lookup(self, normalized: str):
        """Lookup exact match. Returns (prediction, chunk_id) or None."""
        if not normalized:
            return None
        entries = self._index.get(normalized)
        if entries:
            # Return the most recent entry
            return entries[-1][1], entries[-1][0]  # (prediction, chunk_id)
        return None

class SemanticTextCache:
    """L1: Semantic text cache using sentence transformers."""
    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        if SentenceTransformer is None:
            self.encoder = None
            self.index = None
            self.ids = []
            self.predictions = []
            return
        try:
            self.encoder = SentenceTransformer(encoder_name, device=device)
            self.dim = self.encoder.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(self.dim)
            self.ids = []
            self.predictions = []
        except Exception as e:
            print(f"[WARN] Failed to initialize semantic text cache: {e}")
            self.encoder = None
            self.index = None
            self.ids = []
            self.predictions = []
    
    def _encode(self, text: str) -> np.ndarray:
        """Encode text to embedding."""
        if self.encoder is None:
            return None
        embedding = self.encoder.encode([text], normalize_embeddings=True)
        return np.asarray(embedding, dtype="float32")
    
    def add(self, chunk_id: str, text: str, prediction: int):
        """Add entry to semantic cache."""
        if self.encoder is None or self.index is None:
            return
        vec = self._encode(text)
        if vec is not None:
            self.index.add(vec)
            self.ids.append(chunk_id)
            self.predictions.append(prediction)
    
    def search(self, text: str, threshold: float) -> tuple[int, str, float] | None:
        """Search semantic cache. Returns (prediction, chunk_id, score) or None."""
        if self.encoder is None or self.index is None or len(self.ids) == 0:
            return None
        vec = self._encode(text)
        if vec is None:
            return None
        scores, indices = self.index.search(vec, k=1)
        best_score = float(scores[0][0])
        if best_score >= threshold:
            best_idx = int(indices[0][0])
            if 0 <= best_idx < len(self.ids):
                return (self.predictions[best_idx], self.ids[best_idx], best_score)
        return None

def create_empty_cache(embedding_dim, enable_semantic_text=True):
    """Create a fresh empty cache for a new trial. No pre-population, no disk caching."""
    # L0.5: Exact text cache
    exact_cache = ExactTextCache()
    
    # L1: Semantic text cache
    semantic_cache = SemanticTextCache(device=DEVICE) if enable_semantic_text else None
    
    # L2: Embedding cache (vision)
    embedding_index = faiss.IndexFlatIP(embedding_dim)
    cached_labels = np.empty((0,), dtype=np.int64)
    cached_chunk_ids = []  # Track chunk IDs for embedding cache
    
    return exact_cache, semantic_cache, embedding_index, cached_labels, cached_chunk_ids

def predict_with_cache(image, text_prompt, exact_cache, semantic_cache, embedding_index, cached_labels, cached_chunk_ids, 
                       encoder, encoder_transform, alpha_text, alpha_embedding, frame_idx):
    """
    Predict with hierarchical cache lookup following order: L0.5 exact → L1 semantic → L2 embedding.
    Frame N can only access cache from previous frames (0 to N-1) in the same trial.
    Vision embedding (L2) is ALWAYS checked even if text cache (L0.5/L1) hits.
    
    Returns: (prediction, cache_hit_source, exact_cache, semantic_cache, embedding_index, cached_labels, cached_chunk_ids)
    cache_hit_source: "L0.5", "L1", "L2", or None (miss)
    """
    # Normalize text for exact cache
    normalized_text = exact_cache.normalize(text_prompt) if text_prompt else ""
    
    # Extract vision embedding for L2 cache (always done, even if text cache hits)
    vision_emb = get_embeddings([image], encoder, encoder_transform)
    
    # Generate chunk_id for this frame
    chunk_id = f"frame_{frame_idx}"
    
    # Track text cache hits (but we still check L2)
    l05_hit = None
    l1_hit = None
    
    # ===== L0.5: Exact Text Cache =====
    if normalized_text:
        exact_match = exact_cache.lookup(normalized_text)
        if exact_match is not None:
            l05_pred, l05_chunk_id = exact_match
            l05_hit = (l05_pred, l05_chunk_id)
            # Continue to check L1 and L2 even if L0.5 hits
    
    # ===== L1: Semantic Text Cache =====
    if semantic_cache and normalized_text:
        semantic_match = semantic_cache.search(normalized_text, alpha_text)
        if semantic_match is not None:
            l1_pred, l1_chunk_id, l1_score = semantic_match
            l1_hit = (l1_pred, l1_chunk_id, l1_score)
            # Continue to check L2 even if L1 hits
    
    # ===== L2: Embedding Cache (Vision) - ALWAYS checked =====
    if embedding_index.ntotal > 0:
        D, I = embedding_index.search(vision_emb, k=1)
        if D[0][0] > alpha_embedding:
            pred = int(cached_labels[I[0][0]])
            # L2 hit - use this result (vision embedding is most important)
            # Add to all caches for future frames
            exact_cache.add(normalized_text, chunk_id, pred)
            if semantic_cache:
                semantic_cache.add(chunk_id, text_prompt or "", pred)
            embedding_index.add(vision_emb)
            cached_labels = np.append(cached_labels, pred)
            cached_chunk_ids.append(chunk_id)
            return pred, "L2", exact_cache, semantic_cache, embedding_index, cached_labels, cached_chunk_ids
    
    # ===== L2 missed - do full inference =====
    # Even if L0.5 or L1 hit, we do full inference for vision tasks since L2 (vision embedding) is most important
    # We report as miss since we didn't use any cache
    
    # ===== Cache Miss: Full Inference =====
    inputs = feature_extractor(images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    
    # Add to all caches for future frames in this trial only (no disk persistence)
    exact_cache.add(normalized_text, chunk_id, pred)
    if semantic_cache:
        semantic_cache.add(chunk_id, text_prompt or "", pred)
    embedding_index.add(vision_emb)
    cached_labels = np.append(cached_labels, pred)
    cached_chunk_ids.append(chunk_id)
    
    # Report as miss since L2 missed and we did full inference
    # (L0.5/L1 hits are noted but not used for vision classification)
    return pred, None, exact_cache, semantic_cache, embedding_index, cached_labels, cached_chunk_ids

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
# RUN BASELINE (ONCE - doesn't depend on encoder/threshold)
# ===============================
print("\nRunning baseline inference (once for all trials)...")
start_baseline = time.time()
predictions_baseline = []
ground_truth = []
for sample in tqdm(test_dataset, desc="Baseline inference"):
    pred = predict_baseline(sample["image"])
    predictions_baseline.append(pred)
    ground_truth.append(sample["label"])
end_baseline = time.time()
baseline_time = end_baseline - start_baseline
acc_baseline = accuracy_score(ground_truth, predictions_baseline)
print(f"✓ Baseline completed: Accuracy={acc_baseline:.4f}, Time={baseline_time:.2f}s\n")

# ===============================
# RUN ALL TRIALS: ENCODERS × THRESHOLDS
# ===============================
all_results = []

total_trials = len(ENCODERS) * len(THRESHOLDS)
trial_num = 0

for encoder_name in ENCODERS:
    print(f"\n{'='*60}")
    print(f"Testing encoder: {encoder_name}")
    print(f"{'='*60}")
    
    # Load encoder for this trial
    encoder, encoder_transform = load_encoder(encoder_name)
    print(f"✓ Loaded {encoder_name} encoder successfully")
    
    for threshold in THRESHOLDS:
        trial_num += 1
        print(f"\n--- Trial {trial_num}/{total_trials}: {encoder_name} @ threshold {threshold} ---")
        
        # Initialize fresh cache for this trial (no pre-population, no disk caching)
        # Each trial starts with an empty cache - frame N can only access cache from frames 0 to N-1
        dummy_emb = get_embeddings([test_dataset[0]["image"]], encoder, encoder_transform)
        exact_cache, semantic_cache, embedding_index, cached_labels, cached_chunk_ids = create_empty_cache(dummy_emb.shape[1])
        
        predictions_cache, cache_hits = [], []
        cache_hit_sources = []  # Track which cache layer hit
        
        # --- Cache-based inference with hierarchical lookup: L0.5 → L1 → L2 ---
        start_cache = time.time()
        for frame_idx, sample in enumerate(tqdm(test_dataset, desc=f"Cache inference (α={threshold})", leave=False)):
            # For vision-only tasks, text_prompt can be empty, but structure supports it
            text_prompt = ""  # Can be extended to use actual prompts if available
            
            pred, cache_hit_source, exact_cache, semantic_cache, embedding_index, cached_labels, cached_chunk_ids = predict_with_cache(
                sample["image"], 
                text_prompt,
                exact_cache, 
                semantic_cache, 
                embedding_index, 
                cached_labels, 
                cached_chunk_ids,
                encoder, 
                encoder_transform, 
                alpha_text=threshold,  # Threshold for L1 semantic text cache
                alpha_embedding=threshold,  # Threshold for L2 embedding cache
                frame_idx=frame_idx
            )
            predictions_cache.append(pred)
            cache_hits.append(cache_hit_source is not None)
            cache_hit_sources.append(cache_hit_source or "miss")
        end_cache = time.time()
        cache_time = end_cache - start_cache
        
        # Calculate metrics
        acc_cache = accuracy_score(ground_truth, predictions_cache)
        cache_hit_rate = np.mean(cache_hits)
        speedup = baseline_time / cache_time if cache_time > 0 else 0
        
        # Store results
        result = {
            "encoder": encoder_name,
            "threshold": threshold,
            "cache_accuracy": acc_cache,
            "baseline_accuracy": acc_baseline,
            "cache_hit_rate": cache_hit_rate,
            "l05_hits": l05_hits,
            "l1_hits": l1_hits,
            "l2_hits": l2_hits,
            "cache_misses": misses,
            "cache_time": cache_time,
            "baseline_time": baseline_time,
            "speedup": speedup
        }
        all_results.append(result)
        
        # Calculate cache hit breakdown by layer
        l05_hits = cache_hit_sources.count("L0.5")
        l1_hits = cache_hit_sources.count("L1")
        l2_hits = cache_hit_sources.count("L2")
        misses = cache_hit_sources.count("miss")
        
        # Print trial results
        print(f"  Cache Accuracy: {acc_cache:.4f} | Hit Rate: {cache_hit_rate:.4f} | Time: {cache_time:.2f}s | Speedup: {speedup:.2f}×")
        print(f"  Cache breakdown: L0.5={l05_hits}, L1={l1_hits}, L2={l2_hits}, Miss={misses}")
        
        # Explicitly clear cache after trial completion - no disk persistence
        # Cache is only valid within the current trial (frames 0 to N-1)
        del exact_cache, semantic_cache, embedding_index, cached_labels, cached_chunk_ids
        exact_cache = None
        semantic_cache = None
        embedding_index = None
        cached_labels = None
        cached_chunk_ids = None
    
    # Clean up encoder from GPU memory
    del encoder, encoder_transform
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ===============================
# SAVE ALL RESULTS TO CSV
# ===============================
results_df = pd.DataFrame(all_results)
output_file = "experiment1_results.csv"
results_df.to_csv(output_file, index=False)

print(f"\n{'='*60}")
print("=== ALL TRIALS COMPLETE ===")
print(f"{'='*60}")
print(f"\nResults saved to: {output_file}")
print(f"\nSummary of all {total_trials} trials:")
print(results_df.to_string(index=False))

