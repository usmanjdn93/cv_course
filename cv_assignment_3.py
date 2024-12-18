# from PIL import Image, ImageChops


# img1 = Image.open("cv_project/data-gray/walking/frame07.png")
# img2 = Image.open("cv_project/data-gray/walking/frame11.png")

# diff = ImageChops.difference(img1, img2)

# if diff.getbbox():
#     diff.show()
# print(diff.getbbox())

import os
import numpy as np
import cv2
import time

def ssd_algo(img1, img2):
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return np.sum(diff ** 2)

def srd_algo(img1, img2, epsilon=1e-3):
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    rho = np.sqrt(diff ** 2 + epsilon ** 2) - epsilon
    return np.sum(rho)

def sad_algo(img1, img2):
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return np.sum(np.abs(diff))

def process_folder(folder_path):
    # Get all image file paths in the folder
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png'))]

    # Initialize total times for each metric
    total_ssd_time = 0
    total_srd_time = 0
    total_sad_time = 0

    # Run algorithms on every pair of images
    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            img1_path = image_files[i]
            img2_path = image_files[j]

            # Read the images in grayscale
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            # Ensure images have the same dimensions
            if img1.shape != img2.shape:
                print(f"Resizing images: {os.path.basename(img1_path)} and {os.path.basename(img2_path)}")
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # Compute SSD and record time
            start = time.time()
            ssd_algo(img1, img2)
            total_ssd_time += time.time() - start

            # Compute SRD and record time
            start = time.time()
            srd_algo(img1, img2)
            total_srd_time += time.time() - start

            # Compute SAD and record time
            start = time.time()
            sad_algo(img1, img2)
            total_sad_time += time.time() - start

    # Return the total times for the folder
    return {
        "Folder": folder_path,
        "Total SSD Time": total_ssd_time,
        "Total SRD Time": total_srd_time,
        "Total SAD Time": total_sad_time
    }

# Main function to process multiple folders
def process_multiple_folders(base_path):
    """Process multiple folders and collect results."""
    results = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")
            folder_results = process_folder(folder_path)
            if folder_results is not None:
                results.append(folder_results)
    
    return results

# Base directory containing multiple folders
base_path = 'cv_project/data-gray'

# Process all folders
all_results = process_multiple_folders(base_path)

# Print results
print("\nTotal Computation Time per Folder:")
for res in all_results:
    print(f"Folder: {res['Folder']}")
    print(f"  Total SSD Time: {res['Total SSD Time']:.6f}s")
    print(f"  Total SRD Time: {res['Total SRD Time']:.6f}s")
    print(f"  Total SAD Time: {res['Total SAD Time']:.6f}s")
    print("-" * 40)
