import io
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Tuple
from itertools import zip_longest

import numpy as np


def stack_split_images_to_fig(
    image_per_split: Dict[str, np.ndarray],
    split_figsize: Tuple[float, float],
    tight_layout: bool = True,
    stack_vertically: bool = True,
) -> plt.Figure:
    """Stack a mapping of split_name -> split_image into a single figure. This can be done horizontally or vertically.
    :param image_per_split:      Mapping of split_name -> split_image
    :param split_figsize:        Size of each split image. Final size will be a multiple of this, either horizontally or vertically
    :param tight_layout:         Whether to use tight layout
    :param stack_vertically:     Whether to stack the images vertically or horizontally
    :return: Resulting figure
    """
    if stack_vertically:
        fig, axs = plt.subplots(len(image_per_split), 1, figsize=(split_figsize[0], split_figsize[1] * len(image_per_split)))
    else:
        fig, axs = plt.subplots(1, len(image_per_split), figsize=(split_figsize[0] * len(image_per_split), split_figsize[1]))

    for ax, (split, split_images) in zip(axs.flatten(), image_per_split.items()):
        ax.set_axis_off()
        ax.set_title(split)
        ax.imshow(split_images)

    if tight_layout:
        plt.tight_layout()

    return fig


def combine_images(images: List[np.ndarray], n_cols: int, row_figsize: Tuple[float, float], tight_layout: bool = True) -> np.ndarray:
    """Combine a list of images into a single one using matplotlib.
    :param images:              List of images to combine, RGB
    :param n_cols:              Number of images per row
    :param row_figsize:         Figure size of each row. The y-axis will be multiplied by number of rows to determine the overall figsize in y-dim.
    :param tight_layout:        Whether to use tight layout or not
    :return:                    Combined image
    """
    n_rows = len(images) // n_cols + len(images) % n_cols

    fig_size = (row_figsize[0], row_figsize[1] * n_rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip_longest(axs.flatten(), images, fillvalue=None):
        ax.set_axis_off()
        if img is not None:
            ax.imshow(img)

    if tight_layout:
        plt.tight_layout()

    return fig_to_array(fig)


def fig_to_array(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    return np.asarray(image)


def combine_images_per_split_per_class(images_per_split_per_class: Dict[str, Dict[str, np.ndarray]], n_cols: int) -> plt.Figure:
    """For each class, combine split images. Then, combine all the resulting images into one plot.

    Example:
    |------------------------|------------------------|
    | Class1: (train & test) | Class2: (train & test) |
    | Class3: (train & test) | Class4: (train & test) |
    | Class5: (train & test) | Class5: (train & test) |
    |------------------------|------------------------|


    class1:  [train | test]    class2:  [train | test]
    class3:  [train | test]    class4:  [train | test]
    class5:  [train | test]    class6:  [train | test]

    :param images_per_split_per_class:  Mapping of class names and splits to images. e.g. {"class1": {"train": np.ndarray, "valid": np.ndarray},...}
    :param n_cols:                      Number of images per row
    :return:                            Resulting figure
    """
    n_classes = len(images_per_split_per_class)
    n_rows = n_classes // n_cols + n_classes % n_cols

    # Generate one image per class
    images: List[np.ndarray] = []
    for i, (class_name, images_per_split) in enumerate(images_per_split_per_class.items()):

        # This plot is for a single class, which is made of at least 1 split
        class_fig, class_axs = plt.subplots(nrows=1, ncols=len(images_per_split), figsize=(10, 6))
        class_fig.subplots_adjust(top=0.9)
        class_fig.suptitle(f"Class: {class_name}", fontsize=36)

        for (split, split_image), split_ax in zip(images_per_split.items(), class_axs):

            split_ax.imshow(split_image)

            # Write the split name for the first row
            if i < n_cols:
                split_ax.set_xticks([])
                split_ax.set_yticks([])
                split_ax.spines["top"].set_visible(False)
                split_ax.spines["right"].set_visible(False)
                split_ax.spines["bottom"].set_visible(False)
                split_ax.spines["left"].set_visible(False)
                split_ax.set_title(split, fontsize=48)
            else:
                split_ax.set_axis_off()

        class_image = fig_to_array(class_fig)
        images.append(class_image)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 2.5 * n_rows))
    for ax, img in zip_longest(axs.flatten(), images, fillvalue=None):
        ax.axis("off")
        if img is not None:
            ax.imshow(img)
    plt.tight_layout()
    return fig
