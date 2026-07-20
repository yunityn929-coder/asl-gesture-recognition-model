"""
Combined-data training/evaluation for TASK A (camera-degradation augmentation)
and TASK B (ardamavi verified digits 0/1/2/4/5). Additive only — never touches
mlp_model.h5/.tflite, mlp_model_v2_candidate.*, or label_encoder*.pkl.

Data sources merged, each row tagged with a CV "group" and a "source":
  - clean:    X_clean.npy / y_clean.npy / person_ids.npy         group=P1..P10   source=clean
  - degraded: X_degraded.npy / y_degraded.npy / person_ids_degraded.npy
                                                                  group=P1..P10   source=degraded
  - ardamavi: X_ardamavi_digits.npy / y_ardamavi_digits.npy, filtered to
              handshape_verified_ardamavi_digits.npy == True (classes 0,1,2,4,5 only)
                                                                  group=ARDAMAVI_DIGITS  source=ardamavi

Same architecture/training regime as train_ablation.py's winning "raw_eng"
+augmentation variant (BatchNorm->Dense128->Dropout0.3->Dense64->Dropout0.3->
Dense36 softmax, Adam, EarlyStopping patience=20, ReduceLROnPlateau).

Three CV variants, run with --variant:
  degraded_only : 5 person-pair folds (P1+P2 ... P9+P10), train pool =
                  clean+degraded rows of the other 8 persons (ardamavi
                  excluded entirely) -- isolates Task A's effect. Each fold's
                  validation set (both clean and degraded copies of the 2
                  held-out persons) is scored as a whole AND split into
                  clean-only / degraded-only subsets.
  combined      : the same 5 person-pair folds (train pool now also includes
                  all handshape-verified ardamavi rows, since ardamavi is
                  never held out in these folds) PLUS one extra fold that
                  holds out ARDAMAVI_DIGITS entirely (train = all 10 persons'
                  clean+degraded rows) -- 6 folds total, isolates Task B's
                  marginal effect on top of Task A.
  final         : production candidate retrain on ALL rows (clean + degraded
                  + verified ardamavi), 4% stratified monitoring split for
                  EarlyStopping only (not a generalization estimate), saved
                  as mlp_model_v3_candidate.h5/.tflite +
                  label_encoder_v3_candidate.pkl.
"""
import argparse
import json
import os
import numpy as np
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
RANDOM_SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 200

PERSON_PAIRS = [("P1", "P2"), ("P3", "P4"), ("P5", "P6"), ("P7", "P8"), ("P9", "P10")]
ALL_PERSONS = [p for pair in PERSON_PAIRS for p in pair]


def load_all_data():
    X_clean = np.load(os.path.join(BASE_DIR, "X_clean.npy"))
    y_clean = np.load(os.path.join(BASE_DIR, "y_clean.npy"), allow_pickle=True)
    p_clean = np.load(os.path.join(BASE_DIR, "person_ids.npy"), allow_pickle=True)
    src_clean = np.array(["clean"] * len(X_clean), dtype=object)

    X_deg = np.load(os.path.join(BASE_DIR, "X_degraded.npy"))
    y_deg = np.load(os.path.join(BASE_DIR, "y_degraded.npy"), allow_pickle=True)
    p_deg = np.load(os.path.join(BASE_DIR, "person_ids_degraded.npy"), allow_pickle=True)
    src_deg = np.array(["degraded"] * len(X_deg), dtype=object)

    X_ard_all = np.load(os.path.join(BASE_DIR, "X_ardamavi_digits.npy"))
    y_ard_all = np.load(os.path.join(BASE_DIR, "y_ardamavi_digits.npy"), allow_pickle=True)
    verified = np.load(os.path.join(BASE_DIR, "handshape_verified_ardamavi_digits.npy"))
    X_ard, y_ard = X_ard_all[verified], y_ard_all[verified]
    p_ard = np.array(["ARDAMAVI_DIGITS"] * len(X_ard), dtype=object)
    src_ard = np.array(["ardamavi"] * len(X_ard), dtype=object)

    X = np.concatenate([X_clean, X_deg, X_ard], axis=0)
    y = np.concatenate([y_clean, y_deg, y_ard], axis=0)
    group = np.concatenate([p_clean, p_deg, p_ard], axis=0)
    source = np.concatenate([src_clean, src_deg, src_ard], axis=0)

    print(f"[data] clean={len(X_clean)} degraded={len(X_deg)} "
          f"ardamavi_verified={len(X_ard)} (of {len(X_ard_all)} total ardamavi rows) "
          f"total={len(X)}")
    return X, y, group, source


class AugmentedSequence(Sequence):
    def __init__(self, X_raw, y_onehot, augment, batch_size, seed):
        super().__init__()
        self.X_raw = X_raw
        self.y_onehot = y_onehot
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
        eng = lf.compute_engineered_features(X_batch)
        X_batch = np.concatenate([X_batch, eng], axis=1)
        return X_batch.astype(np.float32), self.y_onehot[batch_idx]

    def on_epoch_end(self):
        self.rng.shuffle(self.indices)


def build_features(X_raw):
    eng = lf.compute_engineered_features(X_raw)
    return np.concatenate([X_raw.astype(np.float32), eng], axis=1)


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


def evaluate_subset(model, label_encoder, X_raw, y_raw, tag):
    if len(X_raw) == 0:
        return None
    X_feat = build_features(X_raw)
    y_enc = label_encoder.transform(y_raw)
    y_oh = to_categorical(y_enc, num_classes=len(label_encoder.classes_))
    loss, acc = model.evaluate(X_feat, y_oh, verbose=0)
    y_pred = np.argmax(model.predict(X_feat, verbose=0), axis=1)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    report = classification_report(
        y_raw, y_pred_labels, labels=label_encoder.classes_,
        target_names=label_encoder.classes_, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_raw, y_pred_labels, labels=label_encoder.classes_).tolist()
    return {"tag": tag, "n": int(len(X_raw)), "accuracy": float(acc), "loss": float(loss),
            "per_class_report": report, "confusion_matrix": cm}


def run_fold(X, y, group, source, held_out_groups, include_sources, label_encoder, tag):
    src_mask = np.isin(source, include_sources)
    val_mask = np.isin(group, held_out_groups) & src_mask
    train_mask = ~np.isin(group, held_out_groups) & src_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"[{tag}] held_out={held_out_groups} sources={include_sources} "
          f"train_rows={len(X_train)} val_rows={len(X_val)}")

    num_classes = len(label_encoder.classes_)
    input_dim = lf.RAW_DIM + lf.ENGINEERED_DIM

    y_train_enc = label_encoder.transform(y_train)
    y_train_oh = to_categorical(y_train_enc, num_classes=num_classes)
    y_val_enc = label_encoder.transform(y_val)
    y_val_oh = to_categorical(y_val_enc, num_classes=num_classes)

    model = build_model(input_dim, num_classes)
    train_seq = AugmentedSequence(X_train, y_train_oh, augment=True,
                                   batch_size=BATCH_SIZE, seed=RANDOM_SEED)
    X_val_feat = build_features(X_val)

    early_stopping = EarlyStopping(monitor="val_loss", patience=20,
                                    restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                                   min_lr=1e-6, verbose=0)
    history = model.fit(
        train_seq, validation_data=(X_val_feat, y_val_oh), epochs=MAX_EPOCHS,
        callbacks=[early_stopping, reduce_lr], verbose=2,
    )
    epochs_run = len(history.history["loss"])

    overall = evaluate_subset(model, label_encoder, X_val, y_val, "overall")

    subset_results = {}
    val_source = source[val_mask]
    for src_name in sorted(set(include_sources)):
        sub_mask = val_source == src_name
        res = evaluate_subset(model, label_encoder, X_val[sub_mask], y_val[sub_mask], src_name)
        if res is not None:
            subset_results[src_name] = res

    result = {
        "tag": tag,
        "held_out_groups": list(held_out_groups),
        "include_sources": list(include_sources),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "epochs_run": epochs_run,
        "overall": overall,
        "by_source": subset_results,
    }
    with open(os.path.join(MODEL_DIR, f"summary_{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] overall_acc={overall['accuracy']:.4f} "
          f"by_source={{ {', '.join(f'{k}: {v['accuracy']:.4f}' for k, v in subset_results.items())} }}")
    return result


def cv_degraded_only(X, y, group, source, label_encoder):
    results = []
    for a, b in PERSON_PAIRS:
        tag = f"taskA_degraded_only_{a}{b}"
        res = run_fold(X, y, group, source, held_out_groups=[a, b],
                        include_sources=["clean", "degraded"],
                        label_encoder=label_encoder, tag=tag)
        results.append(res)
    return results


def cv_combined(X, y, group, source, label_encoder):
    results = []
    for a, b in PERSON_PAIRS:
        tag = f"taskAB_combined_{a}{b}"
        res = run_fold(X, y, group, source, held_out_groups=[a, b],
                        include_sources=["clean", "degraded", "ardamavi"],
                        label_encoder=label_encoder, tag=tag)
        results.append(res)
    tag = "taskAB_combined_ARDAMAVI"
    res = run_fold(X, y, group, source, held_out_groups=["ARDAMAVI_DIGITS"],
                    include_sources=["clean", "degraded", "ardamavi"],
                    label_encoder=label_encoder, tag=tag)
    results.append(res)
    return results


def final_retrain(X, y, group, source, label_encoder):
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    num_classes = len(label_encoder.classes_)
    input_dim = lf.RAW_DIM + lf.ENGINEERED_DIM

    train_idx, monitor_idx = train_test_split(
        np.arange(len(X)), test_size=0.04, random_state=RANDOM_SEED, stratify=y)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[monitor_idx], y[monitor_idx]
    print(f"[v3_candidate] FINAL: training on {len(X_train)} rows (all sources), "
          f"{len(X_val)}-row random monitoring split (not a generalization estimate).")

    y_train_oh = to_categorical(label_encoder.transform(y_train), num_classes=num_classes)
    y_val_oh = to_categorical(label_encoder.transform(y_val), num_classes=num_classes)

    model = build_model(input_dim, num_classes)
    train_seq = AugmentedSequence(X_train, y_train_oh, augment=True,
                                   batch_size=BATCH_SIZE, seed=RANDOM_SEED)
    X_val_feat = build_features(X_val)

    early_stopping = EarlyStopping(monitor="val_loss", patience=20,
                                    restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                                   min_lr=1e-6, verbose=0)
    history = model.fit(
        train_seq, validation_data=(X_val_feat, y_val_oh), epochs=MAX_EPOCHS,
        callbacks=[early_stopping, reduce_lr], verbose=2,
    )
    val_loss, val_acc = model.evaluate(X_val_feat, y_val_oh, verbose=0)
    print(f"[v3_candidate] monitor_val_acc={val_acc:.4f} epochs_run={len(history.history['loss'])}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    encoder_path = os.path.join(MODEL_DIR, "label_encoder_v3_candidate.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    model_h5_path = os.path.join(MODEL_DIR, "mlp_model_v3_candidate.h5")
    model.save(model_h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = os.path.join(MODEL_DIR, "mlp_model_v3_candidate.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    summary = {
        "tag": "v3_candidate",
        "sources_included": sorted(set(source.tolist())),
        "train_rows": int(len(X_train)),
        "monitor_val_rows": int(len(X_val)),
        "monitor_val_accuracy_not_a_generalization_estimate": float(val_acc),
        "epochs_run": len(history.history["loss"]),
        "class_labels": list(label_encoder.classes_),
        "model_path": model_h5_path,
        "tflite_path": tflite_path,
        "tflite_size_bytes": os.path.getsize(tflite_path),
        "h5_size_bytes": os.path.getsize(model_h5_path),
    }
    with open(os.path.join(MODEL_DIR, "summary_v3_candidate.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[v3_candidate] saved: {model_h5_path} ({summary['h5_size_bytes']} bytes), "
          f"{tflite_path} ({summary['tflite_size_bytes']} bytes)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["degraded_only", "combined", "final"], required=True)
    args = ap.parse_args()

    X, y, group, source = load_all_data()
    classes_all = sorted(set(y.tolist()))
    label_encoder = LabelEncoder()
    label_encoder.fit(classes_all)
    print(f"[data] classes={list(label_encoder.classes_)} n_classes={len(label_encoder.classes_)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.variant == "degraded_only":
        cv_degraded_only(X, y, group, source, label_encoder)
    elif args.variant == "combined":
        cv_combined(X, y, group, source, label_encoder)
    elif args.variant == "final":
        final_retrain(X, y, group, source, label_encoder)


if __name__ == "__main__":
    main()
