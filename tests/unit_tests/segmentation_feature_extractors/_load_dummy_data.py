from data_gradients.utils import SegmentationBatchData
from data_gradients.preprocess import contours

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_dummy_mask(batch_size: int, image_size: tuple, num_classes: int):
    # Create an empty mask array of the specified size
    np.random.seed(0)
    mask = np.zeros((batch_size, num_classes, image_size[0], image_size[1]), dtype=np.float32)

    # Define the shape parameters
    rect1 = [(10, 10), (60, 160)]
    rect2 = [(25, 25), (125, 150)]
    rect3 = [(75, 75), (225, 125)]

    center1 = (image_size[0] // 2, image_size[1] // 2)
    radius1 = 50

    # Draw shapes in each batch
    for i in range(batch_size):
        # Shift the shapes randomly
        shift1 = np.random.randint(-10, 10, size=2)
        shift2 = np.random.randint(-10, 10, size=2)
        shift3 = np.random.randint(-10, 10, size=2)
        center_shift = np.random.randint(-10, 10, size=2)

        if num_classes >= 1:
            # Draw rectangles in class 0
            cv2.rectangle(mask[i, 0], rect1[0] + shift1, rect1[1] + shift1, 1.0, -1)

        if num_classes >= 2:
            # Draw rectangles in class 1
            cv2.rectangle(mask[i, 1], rect2[0] + shift2, rect2[1] + shift2, 1.0, -1)
            cv2.rectangle(mask[i, 1], rect3[0] + shift3, rect3[1] + shift3, 1.0, -1)

        if num_classes >= 3:
            # Draw circle in class 2
            center_shifted = (center1[0] + center_shift[0], center1[1] + center_shift[1])
            cv2.circle(mask[i, 2], center_shifted, radius1, 1.0, -1)

    # Convert the mask array to a PyTorch tensor
    mask_tensor = torch.from_numpy(mask)

    return mask_tensor


def visualize_dummy_masks(mask_tensor):
    # Get the number of classes and ignore labels
    n_classes = mask_tensor.shape[1]

    # Define a list of colors for each class
    class_colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(n_classes)]

    # Loop over the images in the batch and plot them
    for i in range(mask_tensor.shape[0]):
        # Get the current image from the mask tensor
        image = mask_tensor[i, :, :, :]

        # Initialize an empty RGB image
        rgb_image = np.zeros((image.shape[1], image.shape[2], 3), dtype=np.uint8)

        # Loop over the classes and set the corresponding pixels in the RGB image to the class color
        for class_idx in range(n_classes):
            rgb_image[image[class_idx, :, :] == 1, :] = class_colors[class_idx]

        # Display the RGB image using pyplot
        plt.imshow(rgb_image)
        plt.axis("off")
        plt.show()


def get_dummy_segmentation_batch_data(batch_size: int, height: int, width: int, num_classes: int):
    images = torch.rand(batch_size, 3, height, width)
    labels = generate_dummy_mask(batch_size=batch_size, image_size=(height, width), num_classes=num_classes)
    all_contours = [contours.get_contours(onehot_label) for onehot_label in labels]

    return SegmentationBatchData(images=images, labels=labels, contours=all_contours, split="train")


if __name__ == "__main__":
    batch = get_dummy_segmentation_batch_data(batch_size=5, height=256, width=256, num_classes=5)
    visualize_dummy_masks(batch.labels)
