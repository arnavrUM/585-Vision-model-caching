from datasets import load_dataset, DatasetDict
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from tqdm import tqdm

dataset = load_dataset("Subh775/Rice-Disease-Classification-Dataset")
labels = dataset["train"].features["label"].names
print(dataset, labels)

# debug
# dataset["train"][0]["image"].show()
# print(dataset["train"][0])
# print(labels[dataset["train"][0]["label"]])

split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

model_name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=len(labels), ignore_mismatched_sizes=True)

# TODO: train model on dataset["train"]

def evaluate_model(model, dataset):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for example in tqdm(dataset):
            inputs = processor(example["image"], return_tensors="pt")
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits).item()
            preds.append(pred)
            labels.append(example["label"])
    return np.array(preds), np.array(labels)

preds, labels = evaluate_model(model, dataset["test"])
print("Accuracy:", accuracy_score(labels, preds))
print(confusion_matrix(labels, preds))


