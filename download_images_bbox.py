# Importing the necessary module
import requests
from dotenv import load_dotenv
import os
import os

# Load environment variables from .env file
load_dotenv()


def download_mapillary_images(access_token, bbox, image_size='thumb_2048_url'):
    """
    Downloads images from Mapillary within a specified bounding box.

    :param access_token: Your Mapillary access token.
    :param bbox: The bounding box to search images in (format: 'minLon,minLat,maxLon,maxLat').
    :param image_size: The size of the image to download. Options are 'thumb_256_url', 'thumb_1024_url',
                       'thumb_2048_url', or 'thumb_original_url'. Default is 'thumb_2048_url'.
    """

    # Constructing the URL for image search
    search_url = f'https://graph.mapillary.com/images?access_token={access_token}&fields=id&bbox={bbox}'

    # Sending the request to Mapillary API
    search_response = requests.get(search_url)
    print(f"Status code: {search_response.status_code}")
    image_data = search_response.json()

    # Extracting image IDs from the response
    image_ids = [image['id'] for image in image_data['data']]

    # Create the 'images' directory in the current working directory
    images_dir = os.path.join(os.getcwd(), 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)


    # Looping through each image ID to download images
    for image_id in image_ids:
        # Constructing the URL for each image
        image_url = f"https://graph.mapillary.com/{image_id}?access_token={access_token}&fields={image_size}"

        # Sending request for image details
        image_response = requests.get(image_url)
        image_info = image_response.json()

        # Downloading the image
        img_data = requests.get(image_info[image_size]).content
        with open(os.path.join(images_dir, f'{image_id}.jpg'), 'wb') as handler:
            handler.write(img_data)
            print(f"Downloaded image {image_id}.jpg")

# Example usage
# Set your Mapillary access token and the desired bounding box
access_token = os.getenv('MAPILLARY_ACCESS_TOKEN') # Retrieve the Mapillary access token from the environment variables
bbox = '8.666398776232057, 50.109924204420565, 8.669763052910042, 50.11139420072365' # Example bounding box

# Calling the function to download images
download_mapillary_images(access_token, bbox)
