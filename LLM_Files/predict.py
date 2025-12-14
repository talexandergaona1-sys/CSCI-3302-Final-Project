import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


MODEL_PATH = "./model_output/checkpoint-34888"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


if len(sys.argv) < 2:
    print("To test model type: python predict.py \"PUT TEST TEXT HERE\"")
    sys.exit(1)

text = sys.argv[1]
print(f"\nInput: {text}\n")


inputs = tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=256,
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}


with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

pred_class = int(np.argmax(probs))
suicide_prob = float(probs[1])

label_map = {0: "non-suicide", 1: "suicide"}


print(f"Prediction: {label_map[pred_class].upper()}")
print(f"Suicide Probability: {suicide_prob:.4f}")