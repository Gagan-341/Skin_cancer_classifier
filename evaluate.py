# evaluate.py
import tensorflow as tf
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Load model
model = tf.keras.models.load_model("models/best_model.keras")

# Load validation dataset (same preprocessing as training)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/processed",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=6   # same as training safe batch size
)

# Evaluate model
loss, acc = model.evaluate(val_ds)
print(f"\n✅ Validation Accuracy: {acc*100:.2f}%")
print(f"✅ Validation Loss: {loss:.4f}")

# Get predictions for detailed report
y_true, y_pred = [], []
for batch_x, batch_y in val_ds:
    preds = model.predict(batch_x, verbose=0)
    y_true.extend(batch_y.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("models/confusion_matrix_eval.png")
plt.show()
