"""
DIAGNOSTIC EXPERIMENT — person-disjoint train/validation split.

Mirrors mlp_training.ipynb EXACTLY (same architecture, optimizer, loss,
callbacks) — the ONLY methodology change is the train/val split: instead of
a random row-level train_test_split (which leaks near-duplicate frames of
the same person's hand into both train and val), this holds out entire
persons for validation.

Does NOT touch production files (mlp_model.h5/.tflite, label_encoder.pkl,
training_curves.png, confusion_matrix.png). All outputs use the
"_persondisjoint_test" suffix.

Usage:
    python train_persondisjoint_test.py --held-out P3 P7
    python train_persondisjoint_test.py --auto-select 2   # picks 2 persons automatically
"""
import argparse
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
from collections import Counter

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "model")
RANDOM_SEED = 42
BATCH_SIZE  = 32
MAX_EPOCHS  = 200

SUFFIX = "persondisjoint_test"


def pick_held_out_persons(y, person_ids, n_persons):
    """Greedily choose n_persons whose combined removal leaves every class with
    the most balanced remaining representation, while giving held-out persons
    reasonable per-class coverage. Simple heuristic: pick the n_persons with the
    most total samples (to get a reasonably sized, broad-coverage val set),
    tie-broken by class-coverage breadth."""
    persons = sorted(set(person_ids) - {"UNKNOWN"})
    classes = sorted(set(y))

    coverage = {}
    for p in persons:
        mask = person_ids == p
        classes_present = set(y[mask])
        coverage[p] = (len(classes_present), mask.sum())

    ranked = sorted(persons, key=lambda p: (coverage[p][0], coverage[p][1]), reverse=True)
    return ranked[:n_persons]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", nargs="+", default=None,
                     help="Explicit person ids to hold out for validation, e.g. P3 P7")
    ap.add_argument("--auto-select", type=int, default=None,
                     help="Auto-pick this many persons to hold out")
    args = ap.parse_args()

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    X = np.load(os.path.join(BASE_DIR, "X_clean.npy"))
    y = np.load(os.path.join(BASE_DIR, "y_clean.npy"))
    person_ids = np.load(os.path.join(BASE_DIR, "person_ids.npy"), allow_pickle=True)

    print(f"X shape: {X.shape}, y shape: {y.shape}, person_ids shape: {person_ids.shape}")
    n_unknown = int((person_ids == "UNKNOWN").sum())
    print(f"Rows with unresolved person_id: {n_unknown} ({n_unknown/len(y)*100:.2f}%)")

    if args.held_out:
        held_out = args.held_out
    elif args.auto_select:
        held_out = pick_held_out_persons(y, person_ids, args.auto_select)
    else:
        held_out = pick_held_out_persons(y, person_ids, 2)

    print(f"\nHeld-out persons for validation: {held_out}")

    val_mask = np.isin(person_ids, held_out)
    train_mask = ~val_mask & (person_ids != "UNKNOWN")

    X_train_raw, y_train_raw = X[train_mask], y[train_mask]
    X_val_raw, y_val_raw = X[val_mask], y[val_mask]

    print(f"Train rows: {len(X_train_raw)}  Val rows: {len(X_val_raw)}")

    # Per-class val counts
    classes_all = sorted(set(y))
    val_counts = Counter(y_val_raw)
    train_counts = Counter(y_train_raw)
    print(f"\n{'Class':<8}{'TrainN':>8}{'ValN':>8}")
    low_val_classes = []
    for c in classes_all:
        vn = val_counts.get(c, 0)
        tn = train_counts.get(c, 0)
        flag = " <- LOW/ZERO VAL" if vn < 5 else ""
        if vn < 5:
            low_val_classes.append((c, vn))
        print(f"{c:<8}{tn:>8}{vn:>8}{flag}")

    label_encoder = LabelEncoder()
    label_encoder.fit(classes_all)
    num_classes = len(label_encoder.classes_)

    y_train_enc = label_encoder.transform(y_train_raw)
    y_val_enc = label_encoder.transform(y_val_raw)
    y_train_oh = to_categorical(y_train_enc, num_classes=num_classes)
    y_val_oh = to_categorical(y_val_enc, num_classes=num_classes)

    encoder_path = os.path.join(MODEL_DIR, f"label_encoder_{SUFFIX}.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    model = Sequential([
        BatchNormalization(input_shape=(63,)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ], name="asl_mlp_persondisjoint_test")

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=20,
                                    restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                                   min_lr=1e-6, verbose=1)

    print("\nStarting training (person-disjoint split)...")
    history = model.fit(
        X_train_raw, y_train_oh,
        validation_data=(X_val_raw, y_val_oh),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=2,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history["loss"], label="Train Loss", color="#3498db")
    ax1.plot(history.history["val_loss"], label="Val Loss", color="#e74c3c")
    ax1.set_title("Training Curves (person-disjoint) — Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history.history["accuracy"], label="Train Accuracy", color="#2ecc71")
    ax2.plot(history.history["val_accuracy"], label="Val Accuracy", color="#f39c12")
    ax2.set_title("Training Curves (person-disjoint) — Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(MODEL_DIR, f"training_curves_{SUFFIX}.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    val_loss, val_acc = model.evaluate(X_val_raw, y_val_oh, verbose=0)
    print(f"\n=== FINAL EVALUATION (person-disjoint) ===")
    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

    y_pred = model.predict(X_val_raw, verbose=0)
    y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
    y_true_labels = label_encoder.inverse_transform(np.argmax(y_val_oh, axis=1))

    report_str = classification_report(y_true_labels, y_pred_labels,
                                        labels=label_encoder.classes_,
                                        target_names=label_encoder.classes_,
                                        zero_division=0)
    print(f"\n=== PER-CLASS REPORT (person-disjoint) ===")
    print(report_str)

    report_dict = classification_report(y_true_labels, y_pred_labels,
                                         labels=label_encoder.classes_,
                                         target_names=label_encoder.classes_,
                                         zero_division=0, output_dict=True)

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_encoder.classes_)))
    ax.set_yticks(range(len(label_encoder.classes_)))
    ax.set_xticklabels(label_encoder.classes_)
    ax.set_yticklabels(label_encoder.classes_)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix — Person-Disjoint Split (held out: {held_out})")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, f"confusion_matrix_{SUFFIX}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save model (h5 + tflite) under diagnostic-only filenames
    model_h5_path = os.path.join(MODEL_DIR, f"mlp_model_{SUFFIX}.h5")
    model.save(model_h5_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = os.path.join(MODEL_DIR, f"mlp_model_{SUFFIX}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # Save numeric summary for the report step
    summary = {
        "held_out_persons": held_out,
        "train_rows": int(len(X_train_raw)),
        "val_rows": int(len(X_val_raw)),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "low_val_classes": low_val_classes,
        "per_class_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_labels": list(label_encoder.classes_),
        "epochs_run": len(history.history["loss"]),
    }
    summary_path = os.path.join(MODEL_DIR, f"summary_{SUFFIX}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved (diagnostic only, production untouched):")
    print(f"  {model_h5_path}")
    print(f"  {tflite_path}")
    print(f"  {encoder_path}")
    print(f"  {curves_path}")
    print(f"  {cm_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
