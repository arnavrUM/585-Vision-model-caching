from datasets import load_dataset
from PIL import Image

dataset = load_dataset("Subh775/Rice-Disease-Classification-Dataset")
labels = dataset["train"].features["label"].names
print(dataset, labels)

# debug
# dataset["train"][0]["image"].show()
# print(dataset["train"][0])
# print(labels[dataset["train"][0]["label"]])

