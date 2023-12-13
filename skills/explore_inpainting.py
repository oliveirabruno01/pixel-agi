import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # Load the output JSON file
    with open('inpainting.json', 'r') as f:
        interactions = json.load(f)

    # Select the first interaction
    interaction = interactions[0]

    # Get the palette and image data
    palette = interaction['images'][list(interaction['images'].keys())[0]]['palette'].split('\n')[1:]  # Skip the header
    palette = {line.split(',')[0]: tuple(int(line.split(',')[1][i:i+2], 16) for i in (1, 3, 5, 7)) for line in palette}  # Convert hex color to RGB

    # Create the original image
    image_data = interaction['images'][list(interaction['images'].keys())[0]]['image_data'].split('\n')
    image_data = np.array([list(row) for row in image_data])
    img_array = np.empty((*image_data.shape, 4), dtype=np.uint8)
    for key, color in palette.items():
        img_array[image_data == key] = color
    img = Image.fromarray(img_array)

    # Create the masked image
    action_data = interaction['action'].split('```')[1].strip().split('\n')
    action_data = np.array([list(row) for row in action_data])
    masked_img_array = np.empty((*action_data.shape, 4), dtype=np.uint8)
    for key, color in palette.items():
        masked_img_array[action_data == key] = color
    masked_img = Image.fromarray(masked_img_array)

    # Display the original image and the masked image
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[1].imshow(masked_img)
    axs[1].set_title('Masked Image')
    plt.show()


if __name__ == "__main__":
    main()
