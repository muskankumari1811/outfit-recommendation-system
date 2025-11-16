import os
import numpy as np
import pandas as pd
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.utils import load_img, img_to_array
from tqdm import tqdm
import pickle

# -------------------------------
# 1️⃣ Define paths
# -------------------------------
DATA_DIR = "data"
FEATURES_DIR = "features"

# Create folder to store extracted features
os.makedirs(FEATURES_DIR, exist_ok=True)

# -------------------------------
# 2️⃣ Load pre-trained ResNet50 model
# -------------------------------
print("🧠 Loading ResNet50 model for feature extraction...")
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
print("✅ Model loaded successfully!\n")

# -------------------------------
# 3️⃣ Load metadata (styles.csv)
# -------------------------------
STYLES_PATH = os.path.expanduser(
    "~/.cache/kagglehub/datasets/paramaggarwal/fashion-product-images-small/versions/1/styles.csv"
)

if os.path.exists(STYLES_PATH):
    print("📄 Loading metadata from styles.csv...")
    df = pd.read_csv(STYLES_PATH, on_bad_lines="skip")
    df["image_path"] = "data/" + df["id"].astype(str) + ".jpg"
    print(f"✅ Metadata loaded successfully! {len(df)} records found.\n")
else:
    print("⚠️ styles.csv not found! Please ensure dataset is downloaded correctly.")
    df = pd.DataFrame()

# -------------------------------
# 4️⃣ Feature extraction function
# -------------------------------
def extract_features(img_path):
    """Extract feature vector from an image using ResNet50."""
    try:
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"⚠️ Skipping {img_path}: {e}")
        return None

# -------------------------------
# 5️⃣ Collect all image files
# -------------------------------
image_files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

print(f"📸 Found {len(image_files)} images for feature extraction.\n")

# -------------------------------
# 6️⃣ Extract and store features
# -------------------------------
features_list = []
valid_filenames = []
metadata_records = []

for img_path in tqdm(image_files, desc="Extracting features"):
    feature_vector = extract_features(img_path)
    if feature_vector is not None:
        features_list.append(feature_vector)
        valid_filenames.append(img_path)

        # Match metadata for this image (if available)
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        record = df[df["id"].astype(str) == img_id]
        if not record.empty:
            metadata_records.append(record.iloc[0].to_dict())
        else:
            metadata_records.append({
                "id": img_id,
                "gender": None,
                "masterCategory": None,
                "articleType": None,
                "usage": None,
                "baseColour": None
            })

# -------------------------------
# 7️⃣ Save features, filenames, and metadata
# -------------------------------
features_array = np.array(features_list)
np.save(os.path.join(FEATURES_DIR, "features.npy"), features_array)
pickle.dump(valid_filenames, open(os.path.join(FEATURES_DIR, "filenames.pkl"), "wb"))
pickle.dump(metadata_records, open(os.path.join(FEATURES_DIR, "metadata.pkl"), "wb"))

print("\n✅ Feature extraction completed successfully!")
print(f"💾 Saved feature vectors: {features_array.shape}")
print(f"📁 Files saved in: {FEATURES_DIR}/")
print("🧾 Metadata linked and stored successfully!")
