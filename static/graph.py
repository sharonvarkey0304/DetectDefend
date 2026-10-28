import matplotlib.pyplot as plt
import numpy as np

unsw_stages = ['Baseline', 'Attack', 'Defense']
unsw_accuracy = [98.80, 71.21, 94.71]

plt.figure()
plt.bar(unsw_stages, unsw_accuracy)

plt.title("Model Performance on UNSW-NB15 Dataset")
plt.ylabel("Accuracy (%)")

for i in range(len(unsw_accuracy)):
    plt.text(i, unsw_accuracy[i] + 1, f"{unsw_accuracy[i]}%", ha='center')

plt.savefig("unsw_pipeline.png")
plt.close()


stages = [
    'UNSW Baseline',
    'UNSW Attack',
    'UNSW Defense',
    'NSL-KDD Test'
]

accuracy = [98.80, 71.21, 94.71, 98.80]

plt.figure()
plt.bar(stages, accuracy)

plt.title("Model Performance Across Datasets and Attack Scenarios")
plt.ylabel("Accuracy (%)")

for i in range(len(accuracy)):
    plt.text(i, accuracy[i] + 1, f"{accuracy[i]}%", ha='center')

plt.savefig("accuracy_comparison.png")
plt.close()

metrics = ['Precision', 'Recall', 'F1-Score']
values = [0.9903, 0.9885, 0.9894]

plt.figure()
plt.bar(metrics, values)

plt.title("Performance Metrics on NSL-KDD Dataset")
plt.ylabel("Score")

for i in range(len(values)):
    plt.text(i, values[i] + 0.002, f"{values[i]:.4f}", ha='center')

plt.savefig("nsl_kdd_metrics.png")
plt.close()

cm = np.array([
    [2673, 256],
    [102, 3732]
])

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix (NSL-KDD Dataset)")
plt.colorbar()

labels = ['Normal', 'Attack']
plt.xticks([0, 1], labels)
plt.yticks([0, 1], labels)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.close()
