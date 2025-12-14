import numpy as np
import torch
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
from sklearn.metrics import roc_auc_score

raw_dataset = load_dataset("Ram07/Detection-for-Suicide")

unique_labels = sorted(list(set(raw_dataset["train"]["class"])))
num_labels = len(unique_labels)

label_feature = ClassLabel(
    num_classes=num_labels,
    names=[str(i) for i in unique_labels]
)

raw_dataset = raw_dataset.cast_column("class", label_feature)

dataset_split = raw_dataset["train"].train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="class",
)

train_ds = dataset_split["train"]
val_ds = dataset_split["test"]


def ensure_int_labels(example):
    example["class"] = int(example["class"])
    return example

train_ds = train_ds.map(ensure_int_labels)
val_ds = val_ds.map(ensure_int_labels)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    return tokenizer(
        batch["cleaned_text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)

train_ds = train_ds.rename_column("class", "labels")
val_ds = val_ds.rename_column("class", "labels")

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    preds = np.argmax(probs, axis=1)

    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = float("nan")

    return {"accuracy": acc, "f1": f1, "auc": auc}


training_args = TrainingArguments(
    output_dir="./model_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    weight_decay=0.01,
    logging_steps=500,
    fp16=torch.cuda.is_available(),
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()

results = trainer.evaluate()
print("Final Results: ")
print(results)