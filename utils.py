import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

def build_augmenter():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmenter")

def decode_img(img_bytes, img_size):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [img_size, img_size], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img

def preprocess_path(path, img_size):
    img_bytes = tf.io.read_file(path)
    img = decode_img(img_bytes, img_size)
    return img

def prepare_datasets_from_directory(data_dir, img_size=(224, 224), batch_size=32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    return train_ds, val_ds, class_names

def compute_class_weights(train_ds, num_classes):
    counts = np.zeros((num_classes,), dtype=np.int64)
    for _, y in train_ds.unbatch():
        counts[int(y.numpy())] += 1
    total = counts.sum()
    weights = {i: (total / (num_classes * counts[i])) if counts[i] > 0 else 0.0 for i in range(num_classes)}
    return weights

def load_image_for_inference(image_path, img_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype("float32") 
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_history(history, out_path):
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves")
    plt.savefig(out_path.replace(".png", "_loss.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
