import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


raw = load_dataset("Ram07/Detection-for-Suicide")
full_ds = raw["train"]

def map_labels(ex):
    return {"labels": 1 if ex["class"] == "suicide" else 0}

full_ds = full_ds.map(map_labels)

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    enc = tokenizer(
        batch["cleaned_text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    enc["labels"] = batch["labels"]
    return enc

full_ds = full_ds.map(preprocess, batched=True)

cols_to_keep = ["input_ids", "attention_mask", "labels"]
full_ds = full_ds.remove_columns([c for c in full_ds.column_names if c not in cols_to_keep])

labels_np = np.array(full_ds["labels"])

def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = float("nan")

    return {"accuracy": acc, "f1": f1, "auc": auc}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros_like(labels_np), labels_np), 1):

    train_ds = full_ds.select(train_idx.tolist())
    val_ds = full_ds.select(val_idx.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    args = TrainingArguments(
        output_dir=f"cv_output/fold_{fold_idx}",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=200,
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()


    os.makedirs(f"cv_output/fold_{fold_idx}", exist_ok=True)
    with open(f"cv_output/fold_{fold_idx}/results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    fold_metrics.append(metrics)


avg_acc = np.mean([m["eval_accuracy"] for m in fold_metrics])
avg_f1 = np.mean([m["eval_f1"] for m in fold_metrics])
avg_auc = np.mean([m["eval_auc"] for m in fold_metrics])


print("\n-- Cross Validation Results --")
print(f"Average Accuracy: {avg_acc:.4f}")
print(f"Average F1: {avg_f1:.4f}")
print(f"Average AUC: {avg_auc:.4f}")

with open("cv_output/summary.json", "w") as f:
    json.dump({
        "accuracy": avg_acc,
        "f1": avg_f1,
        "auc": avg_auc,
        "folds": fold_metrics
    }, f, indent=4)