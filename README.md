# outfit-recommendation-system
👗 AI-Powered Outfit Recommender System

An intelligent, deep-learning–powered fashion recommendation system that helps users discover visually and contextually similar outfits based on an uploaded image or selected filters such as gender, category, type, and occasion.

Built using ResNet50, cosine similarity, and a modern Streamlit UI in a stylish dark purple–pink theme.

🌟 Features
🔍 Two Recommendation Modes

Visual Search: Upload an outfit and get similar styling suggestions.

Browse Mode: Get outfit recommendations based on filters without uploading an image.

🎀 Smart Filtering

Filter recommendations by:

Gender

Master Category (Apparel, Accessories, Footwear)

Article Type (Shirts, Dresses, Kurta, Shoes, etc.)

Occasion (Casual, Formal, Sports, Ethnic)

🧠 AI-Based Visual Similarity

Uses ResNet50 (pre-trained on ImageNet) to extract 2048-dimensional feature vectors.

Computes similarity using cosine similarity to retrieve closest matches.

💅 Beautiful Custom UI

Gradient dark purple–pink theme

Clean grid-based layout for recommended outfits

Side-by-side display of uploaded image and results

🗂️ Project Structure
OUTFIT/
│
├── data/                     # Outfit images
├── features/
│   ├── features.npy          # Extracted feature vectors
│   ├── filenames.pkl         # Image file paths
│   ├── metadata.pkl          # Metadata with gender, category, type, etc.
│
├── model.py                  # Feature extraction (ResNet50)
├── recommender.py            # Recommendation logic
├── app.py                    # Streamlit UI
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies list

🧩 Technologies Used

Python 3.10

TensorFlow / Keras

ResNet50

NumPy, Pandas

Scikit-learn

Pillow (PIL)

Streamlit

Matplotlib

🚀 Getting Started
1️⃣ Clone the repository
git clone https://github.com/your-username/outfit-recommender.git
cd outfit-recommender

2️⃣ Create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Download the dataset

Using KaggleHub:

import kagglehub
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")


Copy at least 1000 images into the data/ folder.

5️⃣ Extract features
python model.py

6️⃣ Run the Streamlit app
streamlit run app.py

🎨 User Interface

Upload outfit image

Choose filters (gender, category, type, occasion)

View recommended outfits with captions and similarity score

Works even without image upload

📊 Results

Feature extraction time: ~0.6 sec per image

Similarity computation: < 1 second for 1000 images

Streamlit UI response time: 2–3 seconds

Output: Highly accurate visually similar outfits

🔮 Future Enhancements

Outfit pairing (e.g., match shirt + pants)

Personalized recommendations based on user history

Color palette based matching

Faster search with FAISS / Annoy

Mobile app version

🧑‍💻 Author

Muskan Kumari
B.Tech – Artificial Intelligence & Machine Learning
Manipal University Jaipur
2025
