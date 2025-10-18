
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    ViTForImageClassification,
    ViTFeatureExtractor,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score
import faiss
import pandas as pd

# ========== CONFIGURATION ==========
MODEL_NAME = "google/vit-base-patch16-224-in21k"
DATASET_NAME = "Subh775/Rice-Disease-Classification-Dataset"
OUTPUT_DIR = "./vit_rice_finetuned"
CACHE_THRESHOLD_ALPHA = 0.90  # similarity threshold for caching

# ========== STEP 1: LOAD DATASET ==========
dataset = load_dataset(DATASET_NAME)
labels = dataset["train"].features["label"].names

feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)

def transform(example):
    inputs = feature_extractor(images=example["image"], return_tensors="pt")
    inputs["label"] = example["label"]
    return inputs

dataset = dataset.with_transform(transform)

split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

print(train_dataset)
print(test_dataset)

print(train_dataset[0].keys())

# ========== STEP 2: DEFINE MODEL AND TRAINER ==========
os.environ["WANDB_DISABLED"] = "true"
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels)
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_total_limit=1,
    push_to_hub=False,
    report_to=None,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(OUTPUT_DIR)