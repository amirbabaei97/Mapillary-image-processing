'''This is a basic car detection script that uses a pre-trained Faster R-CNN model.
This is used on a street-level image dataset from Mapillary to detect cars in the images. 
The script is also modified to use the GPU if available.'''

import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torchvision.ops import nms

from PIL import Image, ImageDraw

# Check for GPU availability and set the device accordingly
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {my_device}")

# Load the pre-trained model with the specified weights and move it to the specified device
MODEL = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(my_device)
MODEL.eval()  # Set the model to inference mode

# Function to load images from a directory
def load_images_from_folder(folder):
    '''Loads images from a folder into a list of PIL images'''''
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if not filename.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):
            continue
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        filenames.append(filename)
    return images, filenames

# Function to save an image with its detected bounding boxes
def save_image_with_boxes(image, boxes, filename, output_folder="images_detected"):
    '''Saves an image with its detected bounding boxes'''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    draw = ImageDraw.Draw(image)
    for box in boxes: # Draw bounding boxes around cars
        xmin, ymin, xmax, ymax = box
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red", width=3)
    image.save(os.path.join(output_folder, filename))

# Process each image
def process_images(images, filenames, model, device, detection_threshold=0.5):
    '''Processes images with the model and saves the detected bounding boxes'''
    total_images = len(images)
    for i, (image, filename) in enumerate(zip(images, filenames), start=1):
        image_tensor = F.to_tensor(image).to(device).unsqueeze(0)
        with torch.no_grad():
            prediction = model(image_tensor)

        # Extract boxes, scores, and labels for cars only
        car_boxes = prediction[0]['boxes'][prediction[0]['labels'] == 3]
        car_scores = prediction[0]['scores'][prediction[0]['labels'] == 3]

        # Filter detections by score threshold
        high_conf_indices = car_scores >= detection_threshold
        car_boxes_high_conf = car_boxes[high_conf_indices]
        car_scores_high_conf = car_scores[high_conf_indices]

        # Apply NMS to high confidence detections
        nms_indices = nms(car_boxes_high_conf, car_scores_high_conf, iou_threshold=0.3)
        
        # Filter boxes based on NMS
        nms_boxes = car_boxes_high_conf[nms_indices].cpu().numpy()
        
        # Save image with bounding boxes after NMS and high confidence filtering
        save_image_with_boxes(image, nms_boxes, filename)
        
        print(f"Processing image {i} of {total_images} ({filename})")


# Load images and filenames
Mapillary_images, Mapillary_filenames = load_images_from_folder('images/')

# Process and save images, passing in the device
process_images(Mapillary_images, Mapillary_filenames, MODEL, my_device)
