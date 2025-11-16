import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
from PIL import Image
from recommender import recommend  # reuse your recommender function

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(
    page_title="Outfit Recommender 💜",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Custom Dark Purple-Pink Theme
# -------------------------------
st.markdown("""
    <style>
        body, .stApp {
            background: linear-gradient(135deg, #1f0029, #5f0a87, #a4508b);
            color: #f2e6ff;
            font-family: 'Poppins', sans-serif;
        }

        h1, h2, h3, h4 {
            color: #ffb6d5;
            text-shadow: 0 0 8px #ff4fa1;
        }

        section[data-testid="stSidebar"] {
            background-color: rgba(48, 0, 72, 0.95);
            color: #fff;
            padding: 20px;
        }

        .stButton>button {
            background: linear-gradient(45deg, #8e2de2, #ff0080);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #ff0080, #8e2de2);
            transform: scale(1.05);
        }

        img {
            border-radius: 10px;
            border: 2px solid #ff4fa1;
        }
        .caption {
            text-align: center;
            color: #ffb6d5;
            font-size: 13px;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load metadata
# -------------------------------
@st.cache_resource
def load_metadata():
    metadata = pickle.load(open("features/metadata.pkl", "rb"))
    df = pd.DataFrame(metadata)
    return df

df = load_metadata()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("👗 AI-Powered Outfit Recommender System")
st.markdown("Discover visually or contextually similar outfits by uploading an image **or** just using filters 💫")

# Sidebar filters
st.sidebar.header("🎀 Filter Options")
gender = st.sidebar.selectbox("Select Gender", ["", "Men", "Women", "Boys", "Girls"])
masterCategory = st.sidebar.selectbox("Select Category", ["", "Apparel", "Accessories", "Footwear"])
articleType = st.sidebar.selectbox("Select Type", ["", "Shirts", "Jeans", "Dresses", "Tshirts", "Kurta", "Shoes", "Bags", "Watches"])
usage = st.sidebar.selectbox("Select Occasion", ["", "Casual", "Formal", "Sports", "Ethnic"])
top_n = st.sidebar.slider("Number of recommendations", 3, 10, 5)

# Upload section
uploaded_file = st.file_uploader("📤 Upload an outfit image (optional)", type=["jpg", "jpeg", "png"])

# Layout: Uploaded Image (left) + Recommendations (right)
left_col, right_col = st.columns([1, 3])

results = []

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    uploaded_image = Image.open(uploaded_file)
    with left_col:
        st.image(uploaded_image, caption="Uploaded Outfit", width=280)

    with right_col:
        st.markdown("### 🔍 Finding visually and contextually similar outfits...")
        results = recommend("temp.jpg", gender, masterCategory, articleType, usage, top_n)

else:
    # 🔥 Category-based browsing (no upload)
    st.markdown("### ✨ Showing category-based outfit suggestions...")
    filtered = df.copy()

    if gender:
        filtered = filtered[filtered["gender"].str.contains(gender, case=False, na=False)]
    if masterCategory:
        filtered = filtered[filtered["masterCategory"].str.contains(masterCategory, case=False, na=False)]
    if articleType:
        filtered = filtered[filtered["articleType"].str.contains(articleType, case=False, na=False)]
    if usage:
        filtered = filtered[filtered["usage"].str.contains(usage, case=False, na=False)]

    if not filtered.empty:
        filtered = filtered.sample(min(top_n, len(filtered)))
        results = []
        for _, row in filtered.iterrows():
            img_path = f"data/{row['id']}.jpg"
            if os.path.exists(img_path):
                results.append({
                    "image": img_path,
                    "gender": row.get("gender"),
                    "masterCategory": row.get("masterCategory"),
                    "articleType": row.get("articleType"),
                    "usage": row.get("usage"),
                    "similarity": None
                })

# -------------------------------
# Display Results
# -------------------------------
if results:
    st.markdown("### 💫 Recommended Outfits")
    cols = st.columns(min(len(results), 5))  # limit 5 per row
    for i, rec in enumerate(results):
        if rec["image"] and os.path.exists(rec["image"]):
            with cols[i % 5]:
                st.image(Image.open(rec["image"]), use_column_width=True)
                st.markdown(
                    f"<p class='caption'>{rec['articleType'] or 'Unknown'} | {rec['usage'] or 'N/A'}<br>"
                    f"<b>{rec['gender'] or ''}</b> • {rec['masterCategory'] or ''}"
                    f"<br><span style='color:#ff4fa1;'>{'Similarity: ' + str(rec['similarity']) if rec['similarity'] else ''}</span></p>",
                    unsafe_allow_html=True
                )
else:
    st.warning("No matching outfits found. Try changing filters or uploading another image 😕")
