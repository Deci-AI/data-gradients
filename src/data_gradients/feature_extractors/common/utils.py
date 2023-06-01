from typing import Dict
import numpy as np


class PixelFrequencyCounter:
    """Compute the frequency distribution of pixel intensity values (0-255) for a given channel."""

    def __init__(self):
        self.frequency_dict = {}

    def update(self, image_channel: np.ndarray) -> None:
        """Updates the state with a new image channel."""
        unique_values, counts = np.unique(image_channel, return_counts=True)
        for value, count in zip(unique_values, counts):
            self.frequency_dict[value] = self.frequency_dict.get(value, 0) + count

    def compute(self) -> Dict[int, float]:
        """Compute the frequency per value."""
        return self.frequency_dict
