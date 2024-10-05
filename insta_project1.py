# Project Overview:

# The purpose of this project is to sort through 16,000 screenshots of instagram posts from junk food and health food companies.
# The goals of this project are to detect which images have people in them, count how many people are in them, and detect each person's skin tone.

# File Overview:

# This project has 2 parts.

# Part 1 takes in images and crops each one using edge detection with parts to safeguard against potential edge cases. Screenshots needed cropping because they included comment section information which would negatively influence the results.

# Part 2 is the bulk of the project, which uses Detectron2 to both detect whether a person is in an image and segments them for skin detection. We used k-means clustering for skin-detection to avoid issues from other skin detectors.

# Currently, the project is being improved. We are working on fine-tuning our person detection model by training it on our data, validating our k-means algorithm and selecting an appropriate skin tone scale (and have not used the elbow or silhouette methods yet) and expand our project to include image captioning abilities.

# Part 1: Dynamic Image Cropper

import os
import cv2
import numpy as np

# Helper function to calculate percentage of white pixels
def calculate_white_pixel_percentage(image_path, white_threshold=240):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)

    white_pixel_percentage = np.sum(thresh == 255) / (thresh.shape[0] * thresh.shape[1]) * 100
    return white_pixel_percentage

# Part 1: Cropping pictures by border for one picture
def crop_image_vertically(image_path, output_folder, output_filename):
    try:
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Load the image
        img = cv2.imread(image_path)

        # Check if image was loaded properly
        if img is None:
            print(f"Error: Unable to load image {image_path}")
            return False

        original_height, original_width = img.shape[:2]  # Get original dimensions

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to detect the white background of the comment section
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Sum the pixel values vertically to detect where the image content ends
        vertical_sum = np.sum(thresh, axis=0)

        # Find the first column where the white comment section starts
        comment_start_index = np.argmax(vertical_sum > 0.9 * max(vertical_sum))

        # Crop the image to exclude the right-side comment section
        cropped_img = img[:, :comment_start_index]

        # Check if cropped image is less than 30% of the original image
        cropped_width = cropped_img.shape[1]
        if cropped_width < 0.3 * original_width:
            print(f"Error: Cropped image is less than 30% of the original width. Switching to backup method.")
            return False

        # Save the cropped image
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, cropped_img)
        print(f"Cropped image saved as {output_path}")
        return True

    except cv2.error as e:
        print(f"OpenCV error occurred: {e}")
        return False

# Backup method using line or boundary detection
def save_cropped_image(cropped_img, output_folder, output_filename):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, cropped_img)
    print(f"Cropped image saved as {output_path}")

def detect_vertical_boundary_or_line(image):
    height, width = image.shape[:2]
    right_side = image[:, int(0.45 * width):]

    # Convert to grayscale and apply edge detection
    gray_img = cv2.cvtColor(right_side, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 20, 80)

    # Use Hough Line Transform to detect vertical lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, _, x2, _ = line[0]
            if abs(x2 - x1) < 5:  # Nearly vertical line
                return int(0.45 * width) + x1

    # Fallback: sum along columns to detect vertical boundary if no line found
    vertical_sum = np.sum(edges, axis=0)
    boundary_index = np.argmax(vertical_sum > 0.9 * max(vertical_sum))
    if boundary_index > 0:
        return int(0.45 * width) + boundary_index
    return None

def crop_image_with_backup(image_path, output_folder, output_filename):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return False

    original_height, original_width = img.shape[:2]  # Get original dimensions

    # Try to detect vertical line or boundary and crop the image
    crop_x = detect_vertical_boundary_or_line(img)
    if crop_x:
        print(f"Detected boundary/line at x={crop_x}. Cropping image.")
        cropped_img = img[:, :crop_x]

        # Check if cropped image is less than 30% of the original image
        cropped_width = cropped_img.shape[1]
        if cropped_width < 0.3 * original_width:
            print(f"Error: Cropped image is less than 30% of the original width. Crop failed.")
            return False

        save_cropped_image(cropped_img, output_folder, output_filename)
        return True
    else:
        print("No boundary or vertical line detected.")
        return False

# Function to process images based on white percentage and apply Part 1 or Part 2
def process_images(input_folder, white_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            output_filename = f"cropped_{filename}"

            # Calculate white pixel percentage
            white_percentage = calculate_white_pixel_percentage(image_path)
            print(f"{filename} is {white_percentage:.2f}% white.")

            # Process white images (>90%) with Part 2
            if white_percentage >= 90:
                print(f"Processing {filename} with Part 2 due to high white percentage.")
                crop_image_with_backup(image_path, white_folder, output_filename)
            else:
                # Try Part 1 for non-white images, fallback to Part 2 if needed
                print(f"Processing {filename} with Part 1 (fallback to Part 2 if needed).")
                success = crop_image_vertically(image_path, output_folder, output_filename)
                if not success:
                    crop_image_with_backup(image_path, output_folder, output_filename)

# Example usage
input_folder = '/Users/mattfranco/desktop/junk_random'
white_folder = '/Users/mattfranco/desktop/white_folder'
output_folder = '/Users/mattfranco/desktop/output_folder1'

process_images(input_folder, white_folder, output_folder)

# Part 2: People Detection and Skin Tone Classification

import os
import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from sklearn.cluster import KMeans
import csv
import matplotlib.pyplot as plt

# Path to image directory and output files
image_dir = "/Users/mattfranco/desktop/output_folder1"  # Replace with your directory of images
output_csv1 = "/Users/mattfranco/desktop/output1.csv"
output_csv2 = "/Users/mattfranco/desktop/output2.csv"
output_image = "/Users/mattfranco/desktop/skin_tones_output.jpg"  # Path for output skin tones image

# Define the list of brands (case insensitive)
brands = ["BurgerKing", "ChicFilA", "Chick-Fil-A", "CocaCola", "Dunkin", "McDonalds", "Oreo", "Pepsi", "Pepsico",
          "PizzaHut", "Pizza-Hut", "Redbull", "Starbucks"]


# Function to extract brand from the image name
def extract_brand(image_name):
    for brand in brands:
        if brand.lower() in image_name.lower():
            return brand
    return "Unknown"


# Function to detect skin color using K-Means clustering
def skin_color_kmeans(image_segmented):
    image_segmented_rgb = cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB)
    image_flattened = image_segmented_rgb.reshape((image_segmented_rgb.shape[0] * image_segmented_rgb.shape[1], 3))
    clt = KMeans(n_clusters=4)
    clt.fit(image_flattened)

    def is_black_or_white(color):
        if np.all(color == [0, 0, 0]) or np.all(color == [255, 255, 255]):
            return True
        if np.all(color < [30, 30, 30]) or np.all(color > [240, 240, 240]):
            return True
        return False

    def skin(color):
        temp = np.uint8([[color]])
        color_hsv = cv2.cvtColor(temp, cv2.COLOR_RGB2HSV)
        color_hsv = color_hsv[0][0]
        return (0 <= color_hsv[0] <= 25) and (58 < color_hsv[1] < 174) and (50 <= color_hsv[2] <= 255)

    def centroid_histogram(clt):
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        return hist

    def get_valid_skin_color(hist, centroids):
        for (percent, color) in zip(hist, centroids):
            if not is_black_or_white(color) and skin(color):
                return color
        return None

    hist = centroid_histogram(clt)
    skin_color = get_valid_skin_color(hist, clt.cluster_centers_)
    if skin_color is not None:
        skin_color_hsv = cv2.cvtColor(np.uint8([[skin_color]]), cv2.COLOR_RGB2HSV)[0][0]
        return skin_color_hsv
    else:
        return None


# Function to classify skin types based on HSV values
def classify_skin_type(hsv_value):
    h, s, v = hsv_value  # Unpack HSV values

    # Skin type classification based on HSV ranges
    if h < 10 and s > 100 and v > 100:
        return "Type 1: Very Fair"
    elif h < 20 and s > 100 and v > 100:
        return "Type 2: Fair"
    elif h < 25 and s > 100 and v > 80:
        return "Type 3: Light"
    elif h < 30 and s > 80 and v > 70:
        return "Type 4: Medium"
    elif h < 35 and s > 70 and v > 60:
        return "Type 5: Olive"
    elif h < 40 and s > 60 and v > 50:
        return "Type 6: Brown"
    else:
        return "Type 7: Dark Brown/Black"


# Detectron2 configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

# CSV file initialization
with open(output_csv1, mode='w', newline='') as file1, open(output_csv2, mode='w', newline='') as file2:
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)

    # Write headers
    writer1.writerow(["Image Name", "Contains Person", "Brand"])
    writer2.writerow(["Image Name", "Number of People", "Skin HSV", "Skin Type", "Brand"])

    # Skin tone visualization setup
    skin_tones = []
    skin_type_labels = []

    # Process images
    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):  # Ensure it's an image
            image_path = os.path.join(image_dir, image_name)
            im = cv2.imread(image_path)

            # Perform instance segmentation
            outputs = predictor(im)
            instances = outputs["instances"]
            pred_classes = instances.pred_classes
            person_indices = (pred_classes == 0)
            num_people = person_indices.sum().item()

            # Extract brand from the image name
            brand = extract_brand(image_name)

            # Write to first CSV (image name, contains person, brand)
            writer1.writerow([image_name, num_people > 0, brand])

            # Process second CSV for images containing people
            if num_people > 0:
                person_masks = instances.pred_masks[person_indices].cpu().numpy()
                for i, person_mask in enumerate(person_masks):
                    person_segmented = cv2.bitwise_and(im, im, mask=person_mask.astype(np.uint8))
                    detected_skin_color = skin_color_kmeans(person_segmented)

                    if detected_skin_color is not None:
                        skin_type = classify_skin_type(detected_skin_color)
                        skin_tones.append(detected_skin_color)  # Save for visualization
                        skin_type_labels.append(skin_type)  # Save skin type label
                        writer2.writerow(
                            [f"{image_name}_{chr(65 + i)}", num_people, detected_skin_color, skin_type, brand])
                    else:
                        writer2.writerow(
                            [f"{image_name}_{chr(65 + i)}", num_people, "No valid skin color", "Unknown", brand])

# Visualization of 7 skin types with labels
if skin_tones:
    # Create an image showing the 7 skin tones with labels
    fig, ax = plt.subplots(figsize=(10, 2))
    color_samples = np.zeros((100, 100 * len(skin_tones), 3), dtype=np.uint8)

    for i, hsv_color in enumerate(skin_tones):
        rgb_color = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2RGB)[0][0]
        color_samples[:, i * 100:(i + 1) * 100] = rgb_color

    ax.imshow(color_samples)
    ax.set_xticks([i * 100 + 50 for i in range(len(skin_tones))])
    ax.set_xticklabels(skin_type_labels, rotation=45, ha="right")
    ax.set_yticks([])  # Hide y-axis ticks
    ax.set_title("Detected Skin Tones and Types")

    # Save output image to the desktop
    plt.tight_layout()
    plt.savefig(output_image)
    plt.show()

print(f"Processing complete. CSV files saved and skin tone image saved at {output_image}.")
