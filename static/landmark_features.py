"""
Shared feature-engineering utilities for the HiASL MLP training ablation.

- augment_landmarks: on-the-fly training-time augmentation of the
  normalized (63,) hand-landmark vectors (21 points x xyz, already
  translated to wrist-origin and scaled by wrist-to-middle-MCP distance
  by preprocessing.py's normalize_landmarks). Simulates plausible
  real-world camera-relative hand variation:
    - random in-plane (x,y) rotation, +/-20 deg by default — this is the
      camera-plane rotation a hand undergoes when tilted relative to the
      phone, the dominant real-world source of angle variation. Z is left
      alone: MediaPipe's z is a much noisier relative-depth estimate on a
      different scale than x/y, and rotating it together with x/y would
      mix a reliable signal with an unreliable one.
    - random uniform scale jitter, +/-10% — simulates hand-size/distance
      variation not fully removed by the wrist-to-MCP scale normalization.
    - small Gaussian coordinate noise — simulates landmark-detector jitter.

- compute_engineered_features: derives 17 rotation-invariant shape
  features (finger curl angles + fingertip-to-palm and thumb-to-fingertip
  distances) from the same (63,) vector, for the O/C discriminability
  ablation. Curl angles are invariant to the in-plane rotation above
  (they only depend on relative joint vectors), which raw xyz is not —
  that's the reasoning behind trying them for O vs C vs M, which differ
  mainly in finger curl rather than overall hand orientation.
"""
import numpy as np

NUM_LANDMARKS = 21
RAW_DIM = 63
ENGINEERED_DIM = 17

FINGER_JOINTS = {
    "thumb":  (1, 2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}
PALM_MCPS = (5, 9, 13, 17)
FINGERTIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky


def augment_landmarks(X, rotation_deg=20.0, scale_jitter=0.10, noise_sigma=0.02, rng=None):
    """X: (N, 63) normalized landmark vectors. Returns an augmented copy, (N, 63)."""
    if rng is None:
        rng = np.random.default_rng()
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    pts = X.reshape(n, NUM_LANDMARKS, 3).copy()

    angles = rng.uniform(-rotation_deg, rotation_deg, size=n) * np.pi / 180.0
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    x = pts[:, :, 0].copy()
    y = pts[:, :, 1].copy()
    pts[:, :, 0] = x * cos_a - y * sin_a
    pts[:, :, 1] = x * sin_a + y * cos_a

    scales = rng.uniform(1 - scale_jitter, 1 + scale_jitter, size=n)
    pts *= scales[:, None, None]

    pts += rng.normal(0.0, noise_sigma, size=pts.shape)

    return pts.reshape(n, RAW_DIM).astype(np.float32)


def _angle_between(v1, v2):
    dot = np.sum(v1 * v2, axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos_theta = np.clip(dot / (n1 * n2 + 1e-8), -1.0, 1.0)
    return np.arccos(cos_theta)


def compute_engineered_features(X):
    """X: (N, 63) -> (N, 17): 10 finger-curl angles + 5 fingertip-to-palm
    distances + 2 thumb-to-{index,middle}-tip distances."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    pts = X.reshape(n, NUM_LANDMARKS, 3)
    palm_center = pts[:, PALM_MCPS, :].mean(axis=1)  # (N, 3)

    curl_feats = []
    for joints in FINGER_JOINTS.values():
        j0, j1, j2, j3 = joints
        v_a = pts[:, j1, :] - pts[:, j0, :]
        v_b = pts[:, j2, :] - pts[:, j1, :]
        v_c = pts[:, j3, :] - pts[:, j2, :]
        curl_feats.append(_angle_between(v_a, v_b))
        curl_feats.append(_angle_between(v_b, v_c))
    curl_feats = np.stack(curl_feats, axis=1)  # (N, 10)

    tip_dists = np.stack(
        [np.linalg.norm(pts[:, t, :] - palm_center, axis=-1) for t in FINGERTIPS],
        axis=1,
    )  # (N, 5)

    thumb_index = np.linalg.norm(pts[:, 4, :] - pts[:, 8, :], axis=-1)
    thumb_middle = np.linalg.norm(pts[:, 4, :] - pts[:, 12, :], axis=-1)

    feats = np.concatenate(
        [curl_feats, tip_dists, thumb_index[:, None], thumb_middle[:, None]], axis=1
    )
    assert feats.shape[1] == ENGINEERED_DIM
    return feats.astype(np.float32)
