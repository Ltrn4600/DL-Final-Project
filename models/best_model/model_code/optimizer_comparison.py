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
from tensorflow.keras.optimizers import Adam, SGD


# 1. PROJECT CONFIGURATION
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "uci_har_processed_data.npz"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "optimizer_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

BATCH_SIZE = 64
MAX_EPOCHS = 40

print("=" * 65)
print("HUMAN ACTIVITY RECOGNITION - OPTIMIZER COMPARISON")
print("=" * 65)

# 2. LOAD PREPROCESSED DATA
print("\nLoading preprocessed data...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

data = np.load(DATA_PATH)

X_train = data["X_train"].astype(np.float32)
y_train = data["y_train"].astype(np.int64)

X_val = data["X_val"].astype(np.float32)
y_val = data["y_val"].astype(np.int64)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

assert X_train.shape[1:] == (128, 9)
assert X_val.shape[1:] == (128, 9)

assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()

print("Dataset validation passed.")


# 3. BUILD CNN MODEL
def build_model():

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
            6,
            activation="softmax"
        )
    ])

    return model


# 4. DEFINE EXPERIMENT CONFIGURATIONS
experiments = {
    "Adam": {
        "optimizer": Adam(learning_rate=0.001),
        "model_file": "adam_best.keras"
    },

    "SGD": {
        "optimizer": SGD(
            learning_rate=0.01,
            momentum=0.9
        ),
        "model_file": "sgd_best.keras"
    }
}

results = {}
histories = {}


# 5. TRAIN AND EVALUATE EACH OPTIMIZER
for optimizer_name, config in experiments.items():

    print("\n" + "=" * 65)
    print(f"STARTING EXPERIMENT: {optimizer_name}")
    print("=" * 65)

    # Reset random seeds for reproducibility
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # Build a fresh model
    model = build_model()

    # Compile with selected optimizer
    model.compile(
        optimizer=config["optimizer"],
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\nOptimizer: {optimizer_name}")
    print("\nModel Summary:")
    model.summary()

    model_path = os.path.join(
        OUTPUT_DIR,
        config["model_file"]
    )

    # Define callbacks
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
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        X_train,
        y_train,

        validation_data=(X_val, y_val),

        epochs=MAX_EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        verbose=1
    )

    histories[optimizer_name] = history.history

    # Load the best checkpoint
    best_model = load_model(model_path)

    # Evaluate on validation set only
    val_loss, val_accuracy = best_model.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    best_epoch = (
        int(np.argmin(history.history["val_loss"])) + 1
    )

    results[optimizer_name] = {
        "Best Epoch": best_epoch,
        "Epochs Trained": len(history.history["loss"]),
        "Best Validation Accuracy (%)":
            max(history.history["val_accuracy"]) * 100,
        "Accuracy at Best Loss (%)":
            val_accuracy * 100,
        "Best Validation Loss":
            min(history.history["val_loss"]),
        "Evaluated Validation Loss":
            val_loss
    }

    print("\n" + "-" * 65)
    print(f"{optimizer_name} RESULTS")
    print("-" * 65)

    print(f"Best Epoch: {best_epoch}")

    print(
        f"Best Validation Accuracy: "
        f"{max(history.history['val_accuracy']) * 100:.2f}%"
    )

    print(
        f"Validation Accuracy at Best Loss: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Best Validation Loss: "
        f"{min(history.history['val_loss']):.4f}"
    )


# 6. SAVE COMPARISON RESULTS
print("\n" + "=" * 65)
print("OPTIMIZER COMPARISON RESULTS")
print("=" * 65)

results_df = pd.DataFrame.from_dict(
    results,
    orient="index"
)

print(results_df.round(4))

results_path = os.path.join(
    OUTPUT_DIR,
    "optimizer_comparison_results.csv"
)

results_df.to_csv(results_path)

print("\nComparison results saved:")
print(results_path)


# 7. PLOT ACCURACY COMPARISON
plt.figure(figsize=(10, 6))

for name, history in histories.items():

    plt.plot(
        history["val_accuracy"],
        label=f"{name} Validation Accuracy"
    )

plt.title("Optimizer Comparison - Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "optimizer_accuracy_comparison.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.show()


# 8. PLOT LOSS COMPARISON
plt.figure(figsize=(10, 6))

for name, history in histories.items():

    plt.plot(
        history["val_loss"],
        label=f"{name} Validation Loss"
    )

plt.title("Optimizer Comparison - Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

loss_path = os.path.join(
    OUTPUT_DIR,
    "optimizer_loss_comparison.png"
)

plt.savefig(
    loss_path,
    dpi=300
)

plt.show()


# 9. FINAL SUMMARY
best_optimizer = results_df[
    "Evaluated Validation Loss"
].idxmin()

print("\n" + "=" * 65)
print("OPTIMIZER COMPARISON COMPLETED")
print("=" * 65)

print(f"Better optimizer by validation loss: {best_optimizer}")

print("\nGenerated files:")
print("- optimizer_comparison_results.csv")
print("- optimizer_accuracy_comparison.png")
print("- optimizer_loss_comparison.png")
print("- adam_best.keras")
print("- sgd_best.keras")

print("\nOfficial test data was not used in this experiment.")

print("\nExperiment completed successfully!")
