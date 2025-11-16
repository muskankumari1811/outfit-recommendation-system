import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
import pandas as pd

# -------------------------------
# Load stored features, filenames & metadata
# -------------------------------
print("📦 Loading pre-computed outfit features and metadata...")
feature_list = np.load("features/features.npy")
filenames = pickle.load(open("features/filenames.pkl", "rb"))
metadata = pickle.load(open("features/metadata.pkl", "rb"))
metadata_df = pd.DataFrame(metadata)

print(f"✅ Loaded {len(feature_list)} feature vectors successfully!")
print(f"🧾 Loaded metadata for {len(metadata_df)} outfits.\n")

# -------------------------------
# Load ResNet50 model
# -------------------------------
print("🧠 Loading ResNet50 model for comparison...")
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
print("✅ Model loaded successfully!\n")

# -------------------------------
# Feature extraction function
# -------------------------------
def extract_features(img_path, model):
    """Extract features from uploaded image"""
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten()

# -------------------------------
# Recommendation function with filters
# -------------------------------
def recommend(img_path, gender=None, masterCategory=None, articleType=None, usage=None, top_n=5):
    print(f"🔍 Finding top {top_n} similar outfits for: {img_path}\n")
    input_features = extract_features(img_path, model)
    
    # Compute cosine similarity between input and dataset features
    similarity = cosine_similarity([input_features], feature_list)[0]
    metadata_df["similarity"] = similarity

    # Apply filters (if user specifies)
    filtered_df = metadata_df.copy()
    if gender:
        filtered_df = filtered_df[filtered_df["gender"].str.contains(gender, case=False, na=False)]
    if masterCategory:
        filtered_df = filtered_df[filtered_df["masterCategory"].str.contains(masterCategory, case=False, na=False)]
    if articleType:
        filtered_df = filtered_df[filtered_df["articleType"].str.contains(articleType, case=False, na=False)]
    if usage:
        filtered_df = filtered_df[filtered_df["usage"].str.contains(usage, case=False, na=False)]

    # Sort by similarity score
    filtered_df = filtered_df.sort_values(by="similarity", ascending=False).head(top_n)

    # Collect recommendations
    recommendations = []
    for _, row in filtered_df.iterrows():
        img_path = f"data/{row['id']}.jpg" if "id" in row else None
        recommendations.append({
            "image": img_path,
            "gender": row.get("gender", ""),
            "masterCategory": row.get("masterCategory", ""),
            "articleType": row.get("articleType", ""),
            "usage": row.get("usage", ""),
            "similarity": round(row["similarity"], 3)
        })

    # Print text summary
    print("🧾 Top Recommendations:")
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. {rec['articleType']} ({rec['usage']}) - {rec['gender']} | Similarity: {rec['similarity']}")
    
    return recommendations
