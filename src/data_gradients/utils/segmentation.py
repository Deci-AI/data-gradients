import numpy as np


def mask_to_onehot(mask_categorical: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert a categorical segmentation mask to its one-hot encoded representation.

    :param mask_categorical:    Categorical representation of mask (H, W).
    :param n_classes:           The total number of classes in the dataset.
    :return:                    Onehot representation of mask (C, H, W)
    """
    onehot_mask = np.zeros((n_classes, mask_categorical.shape[0], mask_categorical.shape[1]), dtype=np.int8)

    for c in range(n_classes):
        onehot_mask[c] = mask_categorical == c

    return onehot_mask.astype(np.uint8)
