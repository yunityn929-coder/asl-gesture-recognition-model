"""
Generalized training script for the HiASL MLP fix-pass. Reused for:
  - the raw+augmentation vs raw+engineered-features ablation (single
    P5+P6 holdout),
  - the 5-fold person-disjoint cross-validation of the winning variant,
  - the final production retrain on all 10 persons.

Same base architecture throughout (per the diagnostic task's request —
only the input pipeline changes, never the model):
  BatchNormalization -> Dense(128, relu) -> Dropout(0.3)
                      -> Dense(64, relu)  -> Dropout(0.3)
                      -> Dense(num_classes, softmax)
  Adam, categorical_crossentropy, EarlyStopping(val_loss, patience=20),
  ReduceLROnPlateau(val_loss, factor=0.5, patience=7, min_lr=1e-6).

Two modes:
  Evaluation mode (default): --held-out P5 P6 [...] holds out those
    persons for validation (person-disjoint), reports full metrics.
  Final mode: --final --fixed-epochs N trains on ALL rows (all persons,
    no validation split at all) for exactly N epochs — used only for the
    step-5 production retrain, so as not to withhold any real data from
    the shipped model. N is chosen from the eval-mode k-fold results
    (see cross_validate.py), not tuned via early stopping on held-back data.
"""
import argparse
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

import landmark_features as lf

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "model")
RANDOM_SEED = 42
BATCH_SIZE  = 32
MAX_EPOCHS  = 200


class AugmentedSequence(Sequence):
    """Yields batches with fresh on-the-fly augmentation applied every epoch
    (not a static, one-time augmented copy)."""

    def __init__(self, X_raw, y_onehot, variant, augment, batch_size, seed):
        super().__init__()
        self.X_raw = X_raw
        self.y_onehot = y_onehot
        self.variant = variant
        self.augment = augment
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(X_raw))

    def __len__(self):
        return int(np.ceil(len(self.X_raw) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X_raw[batch_idx]
        if self.augment:
            X_batch = lf.augment_landmarks(X_batch, rng=self.rng)
        if self.variant == "raw_eng":
            eng = lf.compute_engineered_features(X_batch)
            X_batch = np.concatenate([X_batch, eng], axis=1)
        return X_batch.astype(np.float32), self.y_onehot[batch_idx]

    def on_epoch_end(self):
        self.rng.shuffle(self.indices)


def build_features(X_raw, variant):
    """Non-augmented feature build, for validation/eval data."""
    if variant == "raw_eng":
        eng = lf.compute_engineered_features(X_raw)
        return np.concatenate([X_raw.astype(np.float32), eng], axis=1)
    return X_raw.astype(np.float32)


def build_model(input_dim, num_classes):
    model = Sequential([
        BatchNormalization(input_shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ], name="asl_mlp")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def run(held_out, variant, augment, tag, final=False, fixed_epochs=None, verbose=2):
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    X = np.load(os.path.join(BASE_DIR, "X_clean.npy"))
    y = np.load(os.path.join(BASE_DIR, "y_clean.npy"))
    person_ids = np.load(os.path.join(BASE_DIR, "person_ids.npy"), allow_pickle=True)

    classes_all = sorted(set(y))
    label_encoder = LabelEncoder()
    label_encoder.fit(classes_all)
    num_classes = len(label_encoder.classes_)
    input_dim = lf.RAW_DIM + lf.ENGINEERED_DIM if variant == "raw_eng" else lf.RAW_DIM

    if final:
        # Production retrain: use every person's data. A random 4% stratified
        # slice is carved out ONLY to give EarlyStopping/ReduceLROnPlateau a
        # signal to stop on — it is NOT person-disjoint and its accuracy is
        # NOT reported as a generalization estimate (that's the cross-validated
        # number from the k-fold step). This is purely a training-monitoring
        # split, standard practice, and keeps ~96% of all data for training.
        if fixed_epochs is not None:
            train_idx = np.arange(len(X))
            monitor_idx = np.array([], dtype=int)
        else:
            train_idx, monitor_idx = train_test_split(
                np.arange(len(X)), test_size=0.04, random_state=RANDOM_SEED, stratify=y)
        X_train_raw, y_train_raw = X[train_idx], y[train_idx]
        if len(monitor_idx) > 0:
            X_val_raw, y_val_raw = X[monitor_idx], y[monitor_idx]
            print(f"[{tag}] FINAL mode: training on {len(X_train_raw)} rows (all persons), "
                  f"{len(X_val_raw)}-row random monitoring split (not an accuracy estimate).")
        else:
            X_val_raw, y_val_raw = None, None
            print(f"[{tag}] FINAL mode: training on all {len(X_train_raw)} rows, "
                  f"fixed_epochs={fixed_epochs}, no validation split.")
    else:
        val_mask = np.isin(person_ids, held_out)
        train_mask = ~val_mask & (person_ids != "UNKNOWN")
        X_train_raw, y_train_raw = X[train_mask], y[train_mask]
        X_val_raw, y_val_raw = X[val_mask], y[val_mask]
        print(f"[{tag}] held_out={held_out} train_rows={len(X_train_raw)} "
              f"val_rows={len(X_val_raw)}")

    y_train_enc = label_encoder.transform(y_train_raw)
    y_train_oh = to_categorical(y_train_enc, num_classes=num_classes)

    model = build_model(input_dim, num_classes)

    train_seq = AugmentedSequence(X_train_raw, y_train_oh, variant, augment,
                                   BATCH_SIZE, seed=RANDOM_SEED)

    if X_val_raw is None:
        history = model.fit(train_seq, epochs=fixed_epochs, verbose=verbose)
        val_acc = None
        val_loss = None
        report_dict = None
        cm = None
        y_val_oh = None
    else:
        y_val_enc = label_encoder.transform(y_val_raw)
        y_val_oh = to_categorical(y_val_enc, num_classes=num_classes)
        X_val_feat = build_features(X_val_raw, variant)

        early_stopping = EarlyStopping(monitor="val_loss", patience=20,
                                        restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                                       min_lr=1e-6, verbose=0)

        history = model.fit(
            train_seq,
            validation_data=(X_val_feat, y_val_oh),
            epochs=MAX_EPOCHS,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose,
        )

        val_loss, val_acc = model.evaluate(X_val_feat, y_val_oh, verbose=0)
        y_pred = model.predict(X_val_feat, verbose=0)
        y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
        y_true_labels = label_encoder.inverse_transform(np.argmax(y_val_oh, axis=1))

        report_dict = classification_report(
            y_true_labels, y_pred_labels, labels=label_encoder.classes_,
            target_names=label_encoder.classes_, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_).tolist()
        if final:
            print(f"[{tag}] NOTE: the {len(X_val_raw)}-row monitoring-split accuracy "
                  f"below is NOT a generalization estimate (random split, same leakage "
                  f"risk as the original methodology) — it exists only so EarlyStopping "
                  f"has a signal. The honest number is the k-fold cross-validated one.")

        print(f"[{tag}] val_acc={val_acc:.4f} val_loss={val_loss:.4f} "
              f"epochs_run={len(history.history['loss'])}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    encoder_path = os.path.join(MODEL_DIR, f"label_encoder_{tag}.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    model_h5_path = os.path.join(MODEL_DIR, f"mlp_model_{tag}.h5")
    model.save(model_h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = os.path.join(MODEL_DIR, f"mlp_model_{tag}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    has_val = X_val_raw is not None
    fig, ax = plt.subplots(1, 2 if has_val else 1, figsize=(14 if has_val else 7, 5))
    axes = ax if has_val else [ax]
    axes[0].plot(history.history["loss"], label="Train Loss")
    if has_val:
        axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{tag} — Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    if has_val:
        axes[1].plot(history.history["accuracy"], label="Train Acc")
        axes[1].plot(history.history["val_accuracy"], label="Val Acc")
        axes[1].set_title(f"{tag} — Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(MODEL_DIR, f"training_curves_{tag}.png")
    plt.savefig(curves_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "tag": tag,
        "held_out_persons": held_out,
        "variant": variant,
        "augment": augment,
        "final_mode": final,
        "final_monitor_split_only_not_generalization_estimate": bool(final and has_val),
        "train_rows": int(len(X_train_raw)),
        "val_rows": int(len(X_val_raw)) if has_val else None,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "val_accuracy": float(val_acc) if val_acc is not None else None,
        "epochs_run": len(history.history["loss"]),
        "per_class_report": report_dict,
        "confusion_matrix": cm,
        "class_labels": list(label_encoder.classes_),
        "model_path": model_h5_path,
        "tflite_path": tflite_path,
        "tflite_size_bytes": os.path.getsize(tflite_path),
    }
    summary_path = os.path.join(MODEL_DIR, f"summary_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{tag}] saved summary: {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", nargs="*", default=[])
    ap.add_argument("--variant", choices=["raw", "raw_eng"], default="raw")
    ap.add_argument("--augment", action="store_true", default=False)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--final", action="store_true", default=False)
    ap.add_argument("--fixed-epochs", type=int, default=None)
    args = ap.parse_args()

    run(args.held_out, args.variant, args.augment, args.tag,
        final=args.final, fixed_epochs=args.fixed_epochs)


if __name__ == "__main__":
    main()
