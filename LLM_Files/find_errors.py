import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_PATH = "./model_output/checkpoint-34888"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


raw = load_dataset("Ram07/Detection-for-Suicide")

label_map = {"non-suicide": 0, "suicide": 1}

val_ds = raw["train"].train_test_split(test_size=0.2, seed=42)["test"]


original_texts = val_ds["cleaned_text"]


val_ds = val_ds.map(lambda ex: {"labels": label_map[ex["class"]]})


def preprocess(batch):
    enc = tokenizer(
        batch["cleaned_text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    enc["labels"] = batch["labels"]
    return enc

val_ds = val_ds.map(preprocess, batched=True)


val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

loader = torch.utils.data.DataLoader(val_ds, batch_size=32)


false_pos = []
false_neg = []


all_preds = []
all_probs = []

for i, batch in enumerate(loader):
    batch_gpu = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        logits = model(
            input_ids=batch_gpu["input_ids"],
            attention_mask=batch_gpu["attention_mask"]
        ).logits

    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=1)


    preds = preds.cpu().numpy()
    labels = batch["labels"].cpu().numpy()


    for j in range(len(preds)):
        idx = i * loader.batch_size + j

        if idx >= len(original_texts):
            continue  

        text = original_texts[idx]
        pred = preds[j]
        gold = labels[j]

        if pred == 1 and gold == 0:
            false_pos.append(text)
        elif pred == 0 and gold == 1:
            false_neg.append(text)


print("-- False Positives --")
for i, t in enumerate(false_pos[:20]):
    print(f"{i+1}. {t}")

print("-- False Negatives --")
for i, t in enumerate(false_neg[:20]):
    print(f"{i+1}. {t}")

print("\nSummary:")
print(f"False Positives: {len(false_pos)}")
print(f"False Negatives: {len(false_neg)}")