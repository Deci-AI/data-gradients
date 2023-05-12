import dataclasses
from typing import Optional

import pandas as pd


@dataclasses.dataclass
class FeaturesCollection:
    """
    Holds all the extracted features from the dataset
    """

    image_features: pd.DataFrame
    mask_features: Optional[pd.DataFrame]
    bbox_features: Optional[pd.DataFrame]
