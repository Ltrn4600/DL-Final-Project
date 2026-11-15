import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam


# 1. PROJECT CONFIGURATION
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "uci_har_processed_data.npz"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "model_outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("=" * 60)
print("HUMAN ACTIVITY RECOGNITION - MODEL TRAINING")
print("=" * 60)


# 2. LOAD PREPROCESSED DATA
print("\nLoading preprocessed dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Processed dataset not found: {DATA_PATH}"
    )

data = np.load(DATA_PATH)

X_train = data["X_train"].astype(np.float32)
y_train = data["y_train"].astype(np.int64)

X_val = data["X_val"].astype(np.float32)
y_val = data["y_val"].astype(np.int64)

X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"].astype(np.int64)

print("\nDataset shapes:")
print("Training data:", X_train.shape)
print("Training labels:", y_train.shape)

print("Validation data:", X_val.shape)
print("Validation labels:", y_val.shape)

print("Testing data:", X_test.shape)
print("Testing labels:", y_test.shape)


# 3. CHECK DATA
print("\nChecking dataset...")

assert X_train.shape[1:] == (128, 9)
assert X_val.shape[1:] == (128, 9)
assert X_test.shape[1:] == (128, 9)

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)
assert len(X_test) == len(y_test)

assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()
assert np.isfinite(X_test).all()

assert set(np.unique(y_train)).issubset(set(range(6)))
assert set(np.unique(y_val)).issubset(set(range(6)))
assert set(np.unique(y_test)).issubset(set(range(6)))

print("Dataset validation passed.")


# 4. DEFINE ACTIVITY CLASSES
activity_names = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

NUM_CLASSES = len(activity_names)

print("\nActivity classes:")
for i, name in enumerate(activity_names):
    print(f"{i}: {name}")


# 5. BUILD 1D CNN MODEL
print("\nBuilding 1D CNN model...")

model = Sequential([
    Input(shape=(128, 9)),

    Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu",
        padding="same"
    ),

    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    MaxPooling1D(pool_size=2),

    GlobalAveragePooling1D(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# 6. COMPILE MODEL
model.compile(
    optimizer=Adam(learning_rate=0.001),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()


# 7. DEFINE TRAINING CALLBACKS
BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_har_cnn.keras"
)

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.000001,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


# 8. TRAIN MODEL
print("\nStarting model training...")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=40,

    batch_size=64,

    callbacks=callbacks,

    verbose=1
)


# 9. LOAD BEST MODEL
print("\nLoading the best model...")

best_model = load_model(BEST_MODEL_PATH)

print("Best model loaded successfully.")


# 10. EVALUATE ON VALIDATION SET
print("\nEvaluating model on validation data...")

val_loss, val_accuracy = best_model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")


# 11. SAVE TRAINING HISTORY
history_df = pd.DataFrame(history.history)

history_csv_path = os.path.join(
    OUTPUT_DIR,
    "training_history.csv"
)

history_df.to_csv(
    history_csv_path,
    index=False
)

print("\nTraining history saved:")
print(history_csv_path)


# 12. PLOT TRAINING AND VALIDATION ACCURACY
plt.figure(figsize=(9, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "training_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.show()

print("Accuracy plot saved:")
print(accuracy_path)


# 13. PLOT TRAINING AND VALIDATION LOSS
plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

loss_path = os.path.join(
    OUTPUT_DIR,
    "training_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300
)

plt.show()

print("Loss plot saved:")
print(loss_path)


# 14. FINAL SUMMARY
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print(f"Total epochs trained: {len(history.history['loss'])}")

print(f"Best validation accuracy: "
      f"{max(history.history['val_accuracy']) * 100:.2f}%")

print(f"Best validation loss: "
      f"{min(history.history['val_loss']):.4f}")

print(f"Final evaluated validation accuracy: "
      f"{val_accuracy * 100:.2f}%")

print(f"Final evaluated validation loss: "
      f"{val_loss:.4f}")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("- best_har_cnn.keras")
print("- training_history.csv")
print("- training_accuracy.png")
print("- training_loss.png")

print("\nModel training finished successfully!")
