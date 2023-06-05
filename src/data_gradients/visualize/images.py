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
):
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
