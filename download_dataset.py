from pycocotools.coco import COCO
import requests
import os
import pandas as pd

def download_images():
    # Create directories if they don't exist
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)

    # Download from Open Images Dataset
    open_images_url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
    df = pd.read_csv(open_images_url)
    
    # Filter for helmet images
    helmet_images = df[df['LabelName'] == '/m/0zvk5']
    
    # Download images
    for idx, row in helmet_images.iterrows():
        image_url = f"https://storage.googleapis.com/openimages/v6/oidv6-train-images/{row['ImageID']}.jpg"
        response = requests.get(image_url)
        
        if response.status_code == 200:
            with open(f"dataset/images/train/{row['ImageID']}.jpg", 'wb') as f:
                f.write(response.content)

if __name__ == "__main__":
    download_images() 