import kagglehub
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import random
import time

print("🚀 Starting dataset setup for Outfit Recommender...")
time.sleep(0.5)

# Step 1: Download dataset safely
try:
    print("⬇️ Downloading 'Fashion Product Images (Small)' dataset from Kaggle...")
    path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
    print("✅ Dataset downloaded successfully!")
    print("📁 Dataset path:", path)
except Exception as e:
    print("❌ Error while downloading dataset:")
    print(e)
    exit()

# Step 2: Locate images folder
src = os.path.join(path, "images")
if not os.path.exists(src):
    print("❌ Could not find the 'images' folder inside the dataset path.")
    print("Please check that the dataset downloaded correctly.")
    exit()
else:
    print("📸 Found images folder at:", src)

# Step 3: Copy a sample of images to your project 'data/' folder
dest = "data/"
os.makedirs(dest, exist_ok=True)
print("📦 Copying sample images to 'data/' folder...")

all_images = os.listdir(src)
if len(all_images) == 0:
    print("❌ No images found in the dataset folder. Try re-downloading.")
    exit()

# Copy only a subset (1000 images) for speed
sample_size = min(1000, len(all_images))
sample_images = random.sample(all_images, sample_size)

for i, img in enumerate(sample_images):
    shutil.copy(os.path.join(src, img), os.path.join(dest, img))
    if (i + 1) % 100 == 0:
        print(f"   🟢 Copied {i + 1}/{sample_size} images...")

print(f"✅ Successfully copied {sample_size} images to '{dest}' folder!")

# Step 4: Preview a few random images
try:
    print("🖼️ Displaying a few sample images...")
    sample_preview = random.sample(os.listdir(dest), 5)
    plt.figure(figsize=(15, 5))
    for i, img_name in enumerate(sample_preview):
        img_path = os.path.join(dest, img_name)
        img = Image.open(img_path)
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()
    print("✅ Preview displayed successfully.")
except Exception as e:
    print("⚠️ Unable to show images (no GUI available). Saving preview to 'sample_images.png' instead.")
    plt.savefig("sample_images.png")
    print("✅ Saved sample preview as 'sample_images.png'.")
