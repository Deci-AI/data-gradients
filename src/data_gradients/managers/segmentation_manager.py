from typing import Mapping

import pandas as pd
from tqdm import tqdm

from data_gradients.dataset_adapters import SegmentationDatasetAdapter
from data_gradients.feature_extractors import (
    SemanticSegmentationFeaturesExtractor,
    FeaturesResult,
    SegmentationMaskFeatures,
    ImageFeatures,
    ImageFeaturesExtractor,
)
from data_gradients.managers.abstract_manager import AnalysisManagerAbstract


class SegmentationAnalysisManager(AnalysisManagerAbstract):


    @classmethod
    def extract_features(self, dataset: SegmentationDatasetAdapter) -> FeaturesResult:
        image_features_extractor = ImageFeaturesExtractor()
        mask_features_extractor = SemanticSegmentationFeaturesExtractor(ignore_labels=dataset.get_ignored_classes())

        image_features = []
        mask_features = []

        for sample in tqdm(dataset.get_iterator(), desc=f"Extracting features"):
            image_features.append(
                image_features_extractor(sample.image, shared_keys={ImageFeatures.ImageId: sample.sample_id, ImageFeatures.DatasetSplit: "all"})
            )
            mask_features.append(
                mask_features_extractor(
                    sample.mask, shared_keys={SegmentationMaskFeatures.ImageId: sample.sample_id, SegmentationMaskFeatures.DatasetSplit: "all"}
                )
            )

        results = FeaturesResult(
            image_features=pd.concat(map(pd.DataFrame.from_dict, image_features)),
            mask_features=pd.concat(map(pd.DataFrame.from_dict, mask_features)),
            bbox_features=None,
        )

        # This is a bit ugly - after we assembled all the features, we enrich our dataframes with additional columns
        # like class name.
        # The class name helps with plotting to make the plots more readable (E.g. instead of 0,1,2,3 user would see car,plane,truck, etc.)
        results.mask_features[SegmentationMaskFeatures.SegmentationMaskLabelName] = results.mask_features[SegmentationMaskFeatures.SegmentationMaskLabel].apply(
            lambda x: dataset.get_class_names()[x]
        )

        return results

    @classmethod
    def extract_features_from_splits(self, datasets: Mapping[str, SegmentationDatasetAdapter]) -> FeaturesResult:
        image_features = []
        mask_features = []

        for split_name, dataset in datasets.items():
            results = self.extract_features(dataset)
            results.image_features[ImageFeatures.DatasetSplit] = split_name
            results.mask_features[SegmentationMaskFeatures.DatasetSplit] = split_name

            image_features.append(results.image_features)
            mask_features.append(results.mask_features)

        results = FeaturesResult(
            image_features=pd.concat(image_features),
            mask_features=pd.concat(mask_features),
            bbox_features=None,
        )

        return results
