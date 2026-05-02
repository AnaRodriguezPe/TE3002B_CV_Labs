"""
Traffic Sign Classification using SIFT Descriptors
Exercise - Image Classification Pipeline

This script classifies traffic signs using SIFT (Scale-Invariant Feature Transform)
descriptors with FLANN-based matching against reference images.

Signs:
  - AVG Parking (green circular sign)
  - Pedestrians (blue circular sign)
  - Stop (red octagonal sign)

Reference images: parking.jpg, pedestrians.png, stop.jpg (provided as vectorial prints)
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
REFERENCE_IMAGES = {
    "AVG Parking": "parking.jpg",
    "Pedestrians": "pedestrians.png",
    "Stop":        "stop.jpg",
}

PHOTOS_FOLDER = "fotos"
 
# Minimum number of good matches to consider a sign detected
MIN_MATCH_COUNT = 8

# Lowe's ratio test threshold
RATIO_THRESH = 0.75


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load image in BGR; raise FileNotFoundError if missing."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def extract_sift(img_bgr: np.ndarray, sift):
    """Convert to grayscale and extract SIFT keypoints + descriptors."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def match_descriptors(des_ref, des_query, flann):
    """
    FLANN kNN matching with Lowe's ratio test.
    Returns the list of good matches.
    """
    if des_ref is None or des_query is None:
        return []
    if len(des_ref) < 2 or len(des_query) < 2:
        return []

    matches = flann.knnMatch(des_ref, des_query, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_THRESH * n.distance:
                good.append(m)
    return good


def classify_image(query_bgr: np.ndarray, references: dict, sift, flann) -> tuple:
    """
    Classify a query image against all reference sign descriptors.
    Returns (best_label, best_match_count, all_scores).
    """
    kp_q, des_q = extract_sift(query_bgr, sift)

    scores = {}
    for label, (kp_r, des_r) in references.items():
        good = match_descriptors(des_r, des_q, flann)
        scores[label] = len(good)

    best_label = max(scores, key=scores.get)
    best_count = scores[best_label]

    if best_count < MIN_MATCH_COUNT:
        best_label = "Unknown"

    return best_label, best_count, scores


def draw_matches_panel(ref_bgr, kp_r, query_bgr, kp_q, good_matches, label):
    """Return a side-by-side match visualization."""
    vis = cv2.drawMatches(
        ref_bgr, kp_r, query_bgr, kp_q, good_matches[:20], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    h, w = vis.shape[:2]
    cv2.putText(vis, f"Classified as: {label}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Traffic Sign Classification — SIFT Descriptor Pipeline")
    print("=" * 60)

    # ── 1. Initialise SIFT & FLANN ──────────────────────────────
    sift = cv2.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # ── 2. Load & index reference images ───────────────────────
    print("\n[1] Loading reference images …")
    references     = {}   # label → (kp, des)
    ref_images_bgr = {}   # label → bgr array (for visualisation)

    for label, path in REFERENCE_IMAGES.items():
        if not os.path.exists(path):
            print(f"  ✗ Reference missing: {path}  (skipping {label})")
            continue
        img = load_image(path)
        kp, des = extract_sift(img, sift)
        references[label]     = (kp, des)
        ref_images_bgr[label] = img
        print(f"  ✓ {label:20s}  keypoints={len(kp):4d}  path={path}")

    if not references:
        print("ERROR: No reference images found. Exiting.")
        return

    # ── 3. Collect query images from fotos/ ────────────────────
    print(f"\n[2] Scanning query images in '{PHOTOS_FOLDER}/' …")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    photo_paths = sorted(
        p for p in Path(PHOTOS_FOLDER).glob("*")
        if p.suffix.lower() in exts
    )

    if not photo_paths:
        print(f"  No images found in '{PHOTOS_FOLDER}/'. "
              "Generating synthetic test with reference images instead …")
        # Fall back: test against the reference images themselves
        photo_paths = [Path(v) for v in REFERENCE_IMAGES.values() if os.path.exists(v)]

    print(f"  Found {len(photo_paths)} query image(s).")

    # ── 4. Classify each query image ───────────────────────────
    print("\n[3] Classifying …\n")
    print(f"  {'Image':<35} {'Predicted':<20} {'Matches'}")
    print("  " + "-" * 65)

    results          = []   # (path, predicted, scores)
    correct          = 0
    total_with_gt    = 0

    # Ground truth inference from filename prefix (parking/pedestrians/stop)
    gt_map = {
        "parking":     "AVG Parking",
        "pedestrians": "Pedestrians",
        "stop":        "Stop",
    }

    for p in photo_paths:
        try:
            query = load_image(str(p))
        except FileNotFoundError as e:
            print(f"  ! {e}")
            continue

        predicted, best_count, scores = classify_image(query, references, sift, flann)
        results.append((p, predicted, scores))

        # Ground truth (if filename contains a keyword)
        gt = None
        for key, lbl in gt_map.items():
            if key in p.stem.lower():
                gt = lbl
                break

        match_str = "  ".join(f"{k}: {v}" for k, v in scores.items())
        correct_flag = ""
        if gt is not None:
            total_with_gt += 1
            if predicted == gt:
                correct += 1
                correct_flag = "✓"
            else:
                correct_flag = f"✗ (GT={gt})"

        print(f"  {p.name:<35} {predicted:<20} {best_count:3d}  {correct_flag}")

    # ── 5. Accuracy summary ────────────────────────────────────
    print("\n[4] Summary")
    print("  " + "-" * 65)
    if total_with_gt > 0:
        acc = correct / total_with_gt * 100
        print(f"  Accuracy (images with ground-truth label): "
              f"{correct}/{total_with_gt} = {acc:.1f}%")
    else:
        print("  (No ground-truth labels inferred from filenames.)")

    # ── 6. Visualisation ───────────────────────────────────────
    print("\n[5] Generating match visualisation figures …")

    # Show one representative match per sign class
    shown = set()
    fig_rows = []

    for p, predicted, scores in results:
        if predicted == "Unknown" or predicted in shown:
            continue
        shown.add(predicted)

        query = load_image(str(p))
        ref_bgr = ref_images_bgr[predicted]
        kp_r, des_r = references[predicted]
        kp_q, des_q = extract_sift(query, sift)
        good = match_descriptors(des_r, des_q, flann)

        vis = draw_matches_panel(ref_bgr, kp_r, query, kp_q, good, predicted)
        fig_rows.append((predicted, vis))

    if fig_rows:
        n = len(fig_rows)
        fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n))
        if n == 1:
            axes = [axes]
        for ax, (label, vis) in zip(axes, fig_rows):
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Best matches — {label}", fontsize=13, fontweight="bold")
            ax.axis("off")
        plt.tight_layout()
        out_path = "sift_image_matches.png"
        plt.savefig(out_path, dpi=100)
        print(f"  ✓ Saved: {out_path}")
        plt.show()

    # ── 7. Per-class bar chart ─────────────────────────────────
    if results:
        labels_all = list(references.keys())
        # Average match count per true class
        class_scores = {l: [] for l in labels_all}
        for p, predicted, scores in results:
            for key, v in scores.items():
                class_scores[key].append(v)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        x      = np.arange(len(labels_all))
        avgs   = [np.mean(class_scores[l]) if class_scores[l] else 0 for l in labels_all]
        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        bars   = ax2.bar(x, avgs, color=colors, edgecolor="black", linewidth=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_all, fontsize=11)
        ax2.set_ylabel("Average SIFT good matches")
        ax2.set_title("Average SIFT match count per reference class (all query images)")
        for bar, val in zip(bars, avgs):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        out2 = "sift_match_counts.png"
        plt.savefig(out2, dpi=100)
        print(f"  ✓ Saved: {out2}")
        plt.show()

    print("\nDone. ✓")


if __name__ == "__main__":
    main()
