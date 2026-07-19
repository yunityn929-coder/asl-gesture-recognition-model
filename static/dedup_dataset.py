"""
Permanently removes the P5/P6 cross-person duplicate-image contamination
(found in classes I and V, see duplicate_scan_report.json) from the
production dataset arrays.

Works directly on X_clean.npy/y_clean.npy/person_ids.npy: within each
class, groups rows by exact landmark byte value. A group spanning more
than one person_id is a cross-person duplicate (the same image content
saved under two different people) — keep exactly one row per group (the
lowest-numbered person, for determinism), drop the rest.

Within-person duplicate groups (same person, repeated near-identical
frames) are left untouched — that's normal capture redundancy, not the
P5/P6 contamination this script targets.

Backs up the pre-dedup arrays before overwriting, since X_clean.npy/
y_clean.npy are the real dataset files future training runs will load
(unlike the model artifacts, which this task explicitly keeps untouched
until the final step).
"""
import numpy as np
import os
import json
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_PATH = os.path.join(BASE_DIR, "X_clean.npy")
Y_PATH = os.path.join(BASE_DIR, "y_clean.npy")
P_PATH = os.path.join(BASE_DIR, "person_ids.npy")

X_BACKUP = os.path.join(BASE_DIR, "X_clean_prededup_backup.npy")
Y_BACKUP = os.path.join(BASE_DIR, "y_clean_prededup_backup.npy")
P_BACKUP = os.path.join(BASE_DIR, "person_ids_prededup_backup.npy")

REPORT_PATH = os.path.join(BASE_DIR, "dedup_report.json")


def main():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    person_ids = np.load(P_PATH, allow_pickle=True)
    assert len(X) == len(y) == len(person_ids)

    print(f"Loaded: X={X.shape}, y={y.shape}, person_ids={person_ids.shape}")

    if not os.path.exists(X_BACKUP):
        np.save(X_BACKUP, X)
        np.save(Y_BACKUP, y)
        np.save(P_BACKUP, person_ids)
        print(f"Backed up pre-dedup arrays to {X_BACKUP}, {Y_BACKUP}, {P_BACKUP}")
    else:
        print("Backup already exists, not overwriting backup.")

    classes = sorted(set(y))
    drop_indices = set()
    within_person_dup_groups = 0
    cross_person_report = {}

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        groups = defaultdict(list)
        for i in cls_idx:
            groups[X[i].tobytes()].append(i)

        cross_groups_this_class = []
        for key, idxs in groups.items():
            if len(idxs) < 2:
                continue
            persons_here = sorted(set(person_ids[i] for i in idxs), key=lambda p: int(p[1:]))
            if len(persons_here) > 1:
                # cross-person duplicate group: keep the lowest-numbered person's row(s)
                # (if that person contributed more than one identical row, keep just one)
                keep_person = persons_here[0]
                kept = False
                for i in idxs:
                    if person_ids[i] == keep_person and not kept:
                        kept = True
                        continue
                    drop_indices.add(i)
                cross_groups_this_class.append({
                    "persons": persons_here,
                    "group_size": len(idxs),
                    "kept_person": keep_person,
                })
            else:
                within_person_dup_groups += 1

        if cross_groups_this_class:
            cross_person_report[cls] = {
                "cross_person_duplicate_groups": len(cross_groups_this_class),
                "rows_dropped": sum(g["group_size"] - 1 for g in cross_groups_this_class),
                "details": cross_groups_this_class,
            }

    print(f"\nCross-person duplicate contamination found in classes: {list(cross_person_report.keys())}")
    for cls, info in cross_person_report.items():
        print(f"  {cls}: {info['cross_person_duplicate_groups']} groups, "
              f"{info['rows_dropped']} rows dropped")
    print(f"\n(Within-person duplicate groups left untouched: {within_person_dup_groups})")

    keep_mask = np.ones(len(X), dtype=bool)
    keep_mask[list(drop_indices)] = False

    X_new = X[keep_mask]
    y_new = y[keep_mask]
    person_ids_new = person_ids[keep_mask]

    print(f"\nTotal rows dropped: {len(drop_indices)}")
    print(f"New dataset size: {len(X_new)} (was {len(X)})")

    print(f"\n{'Class':<6}{'Before':>8}{'After':>8}{'Dropped':>9}")
    before_counts = Counter(y)
    after_counts = Counter(y_new)
    for cls in classes:
        b = before_counts.get(cls, 0)
        a = after_counts.get(cls, 0)
        print(f"{cls:<6}{b:>8}{a:>8}{b - a:>9}")

    np.save(X_PATH, X_new)
    np.save(Y_PATH, y_new)
    np.save(P_PATH, person_ids_new)
    print(f"\nOverwrote (deduplicated): {X_PATH}, {Y_PATH}, {P_PATH}")

    with open(REPORT_PATH, "w") as f:
        json.dump({
            "total_before": int(len(X)),
            "total_after": int(len(X_new)),
            "total_dropped": len(drop_indices),
            "within_person_dup_groups_untouched": within_person_dup_groups,
            "per_class_before": {c: int(before_counts.get(c, 0)) for c in classes},
            "per_class_after": {c: int(after_counts.get(c, 0)) for c in classes},
            "cross_person_contamination": {
                k: {kk: vv for kk, vv in v.items() if kk != "details"}
                for k, v in cross_person_report.items()
            },
        }, f, indent=2)
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
