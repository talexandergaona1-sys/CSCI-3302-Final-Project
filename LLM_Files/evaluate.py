import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


MODEL_PATH = "./model_output/checkpoint-34888"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


raw = load_dataset("Ram07/Detection-for-Suicide")
dataset = raw["train"].train_test_split(test_size=0.2, seed=42)
val_ds = dataset["test"]


def preprocess(batch):
    enc = tokenizer(
        batch["cleaned_text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    label_map = {"non-suicide": 0, "suicide": 1}
    enc["labels"] = [label_map[x] for x in batch["class"]]
    return enc

val_ds = val_ds.map(preprocess, batched=True)

val_ds = val_ds.remove_columns(
    [col for col in val_ds.column_names if col not in ["input_ids", "attention_mask", "labels"]]
)
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


loader = torch.utils.data.DataLoader(val_ds, batch_size=32)

all_preds = []
all_probs = []
all_labels = []

for batch in loader:
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    all_probs.extend(probs[:, 1].cpu().numpy())
    all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
    all_labels.extend(batch["labels"].cpu().numpy())


all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)


np.save("cv_labels.npy", all_labels)
np.save("cv_probs.npy", all_probs)

print("\nSaved: cv_labels.npy, cv_probs.npy")


acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")

try:
    auc = roc_auc_score(all_labels, all_probs)
except:
    auc = float("nan")


print("\n-- Test Results -- ")
print(f"Accuracy: {acc:.4f}")
print(f"F1: {f1:.4f}")
print(f"AUC: {auc:.4f}")