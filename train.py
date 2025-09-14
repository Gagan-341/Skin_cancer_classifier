import argparse
import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import plot_confusion_matrix, plot_history
import psutil

print("⚠️ Training forced on CPU. GPU will not be used.")
tf.config.set_visible_devices([], 'GPU')  # force CPU

def build_model(input_shape, num_classes, model_type="efficientnet"):
    if model_type.lower() == "mobilenet":
        base_model = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        preprocess = keras.applications.mobilenet_v2.preprocess_input
    else:  # EfficientNetB0
        base_model = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        preprocess = keras.applications.efficientnet.preprocess_input

    base_model.trainable = False
    inputs = keras.Input(shape=input_shape)
    x = preprocess(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def get_safe_batch_size(desired_batch=32):
    """Estimate safe batch size from available RAM"""
    mem = psutil.virtual_memory()
    free_gb = mem.available / (1024**3)
    # heuristic: allow batch size proportional to free RAM
    safe_batch = max(1, int(desired_batch * min(1, free_gb / 8)))
    print(f"Estimated safe batch size from RAM: {safe_batch}")
    return safe_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_type", type=str, default="mobilenet",
                        choices=["mobilenet", "efficientnet"])
    args = parser.parse_args()

    batch_size = get_safe_batch_size(args.batch_size)
    img_size = (224, 224)

    print(f"\n🔹 Loading datasets from {args.data_dir} ...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print("✅ Classes:", class_names)
    os.makedirs("models", exist_ok=True)
    with open("models/class_names.json", "w") as f:
        json.dump(class_names, f)

    # --- Performance ---
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # --- Model ---
    print(f"\n🚀 Building {args.model_type} model...")
    model = build_model(input_shape=(224, 224, 3), num_classes=len(class_names), model_type=args.model_type)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --- Callbacks ---
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "models/best_model.keras", save_best_only=True
    )
    earlystop_cb = keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, monitor='val_loss'
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=3, verbose=1, monitor='val_loss'
    )

    print(f"\n🚀 Starting training for {args.epochs} epochs with batch size {batch_size} ...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
        )
    except tf.errors.ResourceExhaustedError:
        print("\n⚠️ Out of memory! Try lowering batch size manually (e.g., --batch_size 8)")
        return

    # --- Save final model ---
    model.save("models/final_model.keras")
    plot_history(history, "models/history.png")

    # --- Confusion matrix ---
    from sklearn.metrics import confusion_matrix
    import numpy as np

    y_true, y_pred = [], []
    for batch_x, batch_y in val_ds:
        preds = model.predict(batch_x, verbose=0)
        y_true.extend(batch_y.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, "models/confusion_matrix.png")
    print("\n✅ Training complete. Models saved in 'models/'")


if __name__ == "__main__":
    main()
