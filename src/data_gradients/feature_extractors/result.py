import dataclasses
from typing import Optional

import pandas as pd

@dataclasses.dataclass
class FeaturesResult:
    image_features: pd.DataFrame
    mask_features: Optional[pd.DataFrame]
    bbox_features: Optional[pd.DataFrame]
