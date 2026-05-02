import cv2
import numpy as np
import os
from pathlib import Path

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────
DOWNLOADS = Path.home() / "Downloads"
PHOTOS_FOLDER = DOWNLOADS / "fotos"

RATIO_THRESH = 0.75
MIN_MATCH_COUNT = 15
FRAME_SKIP = 3

# Mapeo de nombres
LABEL_MAP = {
    "parking": "AVG Parking",
    "stop": "Stop",
    "pedestrians": "Pedestrians",
}

# ─────────────────────────────────────────────
# Funciones
# ─────────────────────────────────────────────
def resize_image(img, max_size=500):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def extract_sift(img, sift):
    img = resize_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def match_descriptors(des1, des2, flann):
    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < RATIO_THRESH * n.distance:
            good.append(m)

    return good

def classify(des_q, references, flann):
    scores = {}

    for label, descriptors_list in references.items():
        total_matches = 0

        for des_r in descriptors_list:
            good = match_descriptors(des_r, des_q, flann)
            total_matches += len(good)

        scores[label] = total_matches

    # ordenar
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_label, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

    # reglas estrictas
    if best_score < MIN_MATCH_COUNT:
        return "Unknown", best_score

    if best_score < second_score * 1.2:
        return "Unknown", best_score

    return best_label, best_score

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Cargando dataset desde fotos...")

    sift = cv2.SIFT_create(nfeatures=300)

    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=20)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # ── 1. Cargar dataset automáticamente ──
    references = {}

    exts = {".jpg", ".jpeg", ".png"}

    for img_path in PHOTOS_FOLDER.glob("*"):
        if img_path.suffix.lower() not in exts:
            continue

        name = img_path.stem.lower()

        # detectar clase por nombre
        label = None
        for key in LABEL_MAP:
            if key in name:
                label = LABEL_MAP[key]
                break

        if label is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        kp, des = extract_sift(img, sift)

        if des is None:
            continue

        if label not in references:
            references[label] = []

        references[label].append(des)

        print(f"{img_path.name} → {label} ({len(kp)} kp)")

    if not references:
        print("No se encontraron imágenes válidas.")
        return

    print("\nClases cargadas:")
    for k, v in references.items():
        print(f"{k}: {len(v)} imágenes")

    # ── 2. Cámara ──
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    print("\nPresiona 'q' para salir")

    frame_count = 0
    last_label = "..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            kp_q, des_q = extract_sift(frame, sift)

            label, score = classify(des_q, references, flann)
            last_label = f"{label} ({score})"

        # dibujar resultado
        cv2.putText(frame, last_label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Detección de Señales", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
