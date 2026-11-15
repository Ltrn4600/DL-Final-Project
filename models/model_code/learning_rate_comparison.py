import os
import random
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# 1. CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "uci_har_processed_data.npz"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "learning_rate_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 40

LEARNING_RATES = [0.0001, 0.001, 0.01]

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


# 2. SET RANDOM SEEDS
def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# 3. LOAD PREPROCESSED DATA
print("=" * 65)
print("HUMAN ACTIVITY RECOGNITION - LEARNING RATE COMPARISON")
print("=" * 65)

print("\nLoading preprocessed data...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Preprocessed data not found: {DATA_PATH}"
    )

data = np.load(DATA_PATH)

X_train = data["X_train"].astype("float32")
y_train = data["y_train"].astype("int32")

X_val = data["X_val"].astype("float32")
y_val = data["y_val"].astype("int32")

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

if X_train.ndim != 3 or X_train.shape[1:] != (128, 9):
    raise ValueError("Unexpected training data shape.")

if X_val.ndim != 3 or X_val.shape[1:] != (128, 9):
    raise ValueError("Unexpected validation data shape.")

if not np.all(np.isfinite(X_train)):
    raise ValueError("Training data contains NaN or Inf.")

if not np.all(np.isfinite(X_val)):
    raise ValueError("Validation data contains NaN or Inf.")

print("Dataset validation passed.")


# 4. BUILD CNN MODEL
def build_cnn_model(learning_rate):

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

        Dense(64, activation="relu"),

        Dropout(0.3),

        Dense(6, activation="softmax")
    ])

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# 5. TRAIN MODELS WITH DIFFERENT LEARNING RATES
results = {}
histories = {}

for learning_rate in LEARNING_RATES:

    tf.keras.backend.clear_session()
    set_random_seeds(RANDOM_SEED)

    experiment_name = f"lr_{learning_rate}"

    print("\n" + "=" * 65)
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print("=" * 65)

    model = build_cnn_model(learning_rate)

    print("\nModel Summary:")
    model.summary()

    model_path = os.path.join(
        OUTPUT_DIR,
        f"{experiment_name}_best.keras"
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    # Save training history
    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{experiment_name}_history.csv"
        ),
        index=False
    )

    histories[learning_rate] = history.history

    # Load the best model based on validation loss
    best_model = load_model(model_path)

    val_loss, val_accuracy = best_model.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    best_epoch = (
        int(np.argmin(history.history["val_loss"])) + 1
    )

    best_val_loss = float(
        min(history.history["val_loss"])
    )

    best_val_accuracy = float(
        max(history.history["val_accuracy"])
    )

    accuracy_at_best_loss = float(
        history.history["val_accuracy"][best_epoch - 1]
    )

    results[learning_rate] = {
        "Best Epoch": best_epoch,
        "Best Validation Accuracy": best_val_accuracy,
        "Accuracy at Best Loss": accuracy_at_best_loss,
        "Best Validation Loss": best_val_loss,
        "Evaluated Validation Accuracy": float(val_accuracy),
        "Evaluated Validation Loss": float(val_loss),
        "Training Epochs": len(history.history["loss"])
    }

    print("\n" + "-" * 65)
    print(f"RESULTS FOR LEARNING RATE: {learning_rate}")
    print("-" * 65)

    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Accuracy at Best Loss: {accuracy_at_best_loss * 100:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Evaluated Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Evaluated Validation Loss: {val_loss:.4f}")


# 6. SAVE COMPARISON RESULTS
results_df = pd.DataFrame(results).T

results_df.index.name = "Learning Rate"

results_csv = os.path.join(
    OUTPUT_DIR,
    "learning_rate_comparison_results.csv"
)

results_df.to_csv(results_csv)

print("\n" + "=" * 65)
print("LEARNING RATE COMPARISON RESULTS")
print("=" * 65)

print(results_df.to_string())


# 7. PLOT VALIDATION ACCURACY
plt.figure(figsize=(10, 6))

for learning_rate in LEARNING_RATES:

    accuracy = histories[learning_rate]["val_accuracy"]

    plt.plot(
        range(1, len(accuracy) + 1),
        accuracy,
        marker="o",
        markersize=3,
        label=f"Learning Rate = {learning_rate}"
    )

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

accuracy_plot = os.path.join(
    OUTPUT_DIR,
    "learning_rate_accuracy_comparison.png"
)

plt.savefig(accuracy_plot, dpi=300)
plt.close()


# 8. PLOT VALIDATION LOSS
plt.figure(figsize=(10, 6))

for learning_rate in LEARNING_RATES:

    loss = histories[learning_rate]["val_loss"]

    plt.plot(
        range(1, len(loss) + 1),
        loss,
        marker="o",
        markersize=3,
        label=f"Learning Rate = {learning_rate}"
    )

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

loss_plot = os.path.join(
    OUTPUT_DIR,
    "learning_rate_loss_comparison.png"
)

plt.savefig(loss_plot, dpi=300)
plt.close()


# 9. IDENTIFY THE BEST LEARNING RATE
best_learning_rate = results_df[
    "Evaluated Validation Loss"
].idxmin()

print("\n" + "=" * 65)
print("LEARNING RATE EXPERIMENT COMPLETED")
print("=" * 65)

print(
    "Best learning rate based on validation loss:",
    best_learning_rate
)

print("\nGenerated files:")

print("- learning_rate_comparison_results.csv")
print("- learning_rate_accuracy_comparison.png")
print("- learning_rate_loss_comparison.png")
print("- Best model files for all three learning rates")
print("- Individual training history CSV files")

print("\nOfficial test data was not used in this experiment.")
print("\nExperiment completed successfully!")
