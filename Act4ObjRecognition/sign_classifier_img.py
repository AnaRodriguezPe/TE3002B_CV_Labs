import cv2
import os
import numpy as np

# --------------------------
# CONFIGURATION
# --------------------------
folder = "fotos"

classes = {
    "stop": [],
    "pedestrian": [],
    "parking": []
}

# Test images outside training folder
tests = [
    ("stop_test.png", "stop"),
    ("pedestrian_test.png", "pedestrian"),
    ("parking_test.png", "parking")
]

# --------------------------
# LOAD TRAINING IMAGES
# --------------------------
print("Loading training images...\n")

for file in os.listdir(folder):
    path = os.path.join(folder, file)

    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img, (350, 350))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if file.startswith("stop"):
        classes["stop"].append(gray)

    elif file.startswith("pedestrian"):
        classes["pedestrian"].append(gray)

    elif file.startswith("parking"):
        classes["parking"].append(gray)

for label in classes:
    print(f"{label}: {len(classes[label])} training images loaded")

# --------------------------
# CREATE SIFT
# --------------------------
print("\nCreating SIFT detector...")
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# --------------------------
# EXTRACT DESCRIPTORS
# --------------------------
print("Extracting descriptors...\n")

database = {}

for label in classes:
    database[label] = []

    for img in classes[label]:
        kp, des = sift.detectAndCompute(img, None)

        if des is not None:
            database[label].append(des)

    print(f"{label}: {len(database[label])} descriptor sets stored")

# --------------------------
# CLASSIFICATION FUNCTION
# --------------------------
def classify_image(filename, expected_label):

    print("\n----------------------------------")
    print(f"Testing image: {filename}")
    print(f"Expected class: {expected_label}")

    test = cv2.imread(filename)

    if test is None:
        print("Image not found.")
        return

    test = cv2.resize(test, (350, 350))
    gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    kp_test, des_test = sift.detectAndCompute(gray_test, None)

    if des_test is None:
        print("No descriptors found.")
        return

    best_label = "Unknown"
    best_score = 0

    for label in database:

        total_matches = 0

        for des_ref in database[label]:

            matches = bf.knnMatch(des_ref, des_test, k=2)

            good = []

            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            total_matches += len(good)

        print(f"Matches with {label}: {total_matches}")

        if total_matches > best_score:
            best_score = total_matches
            best_label = label

    # Result
    print(f"\nPredicted class: {best_label}")
    print(f"Best score: {best_score}")

    if best_label == expected_label:
        print("Result: CORRECT")
        color = (0, 255, 0)
    else:
        print("Result: INCORRECT")
        color = (0, 0, 255)

    # Show image
    cv2.putText(test,
                f"Predicted: {best_label}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2)

    cv2.putText(test,
                f"Expected: {expected_label}",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2)

    cv2.imshow("Classification Result", test)
    cv2.waitKey(0)

# --------------------------
# RUN ALL TESTS
# --------------------------
print("\nStarting automatic tests...")

correct = 0

for filename, expected in tests:
    classify_image(filename, expected)

cv2.destroyAllWindows()

print("\nAll tests completed.")
