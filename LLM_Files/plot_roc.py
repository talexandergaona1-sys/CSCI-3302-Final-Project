import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


labels = np.load("cv_labels.npy")
probs  = np.load("cv_probs.npy")


fpr, tpr, thresholds = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positives')
plt.ylabel('True Positives')
plt.title('ROC Curve for Suicide Detection Model')
plt.legend(loc="lower right")

plt.grid(True)

plt.show()