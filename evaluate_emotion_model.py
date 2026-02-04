import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load emotion classifier
print("Loading emotion classifier...")
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)
print("Model loaded.")

# Load test dataset
data = pd.read_csv("emotion_test_data.csv")

y_true = []
y_pred = []

# Run predictions
for text, true_label in zip(data["text"], data["true_emotion"]):
    result = classifier(text)[0]
    scores = {item["label"].lower(): item["score"] for item in result}
    predicted_label = max(scores, key=scores.get)

    y_true.append(true_label)
    y_pred.append(predicted_label)

# ✅ Accuracy Metrics
print("\n=== Accuracy Score ===")
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", round(accuracy, 3))

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred))

# ✅ Confusion Matrix
labels = sorted(list(set(y_true)))
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Emotion Classification Confusion Matrix")
plt.show()
