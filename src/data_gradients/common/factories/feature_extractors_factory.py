from .base_factory import BaseFactory
from ..registry.registry import FEATURE_EXTRACTORS


class FeatureExtractorsFactory(BaseFactory):
    def __init__(self):
        super().__init__(FEATURE_EXTRACTORS)

