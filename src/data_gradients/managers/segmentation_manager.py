import itertools
from multiprocessing import Pool
from typing import Mapping

import pandas as pd
from tqdm import tqdm

from data_gradients.dataset_adapters import SegmentationDatasetAdapter, SegmentationSample
from data_gradients.feature_extractors import (
    SemanticSegmentationFeaturesExtractor,
    FeaturesCollection,
    SegmentationMaskFeatures,
    ImageFeatures,
    ImageFeaturesExtractor,
)
from data_gradients.writers import NotebookWriter
from data_gradients.managers.abstract_manager import AnalysisManagerAbstract
from data_gradients.reports.report_template import ReportTemplate


class DummyPool:
    def __init__(self, num_workers):
        if num_workers > 0:
            raise ValueError("DummyPool can't have workers")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


class SegmentationAnalysisManager(AnalysisManagerAbstract):
    @classmethod
    def process_sample(cls, sample: SegmentationSample, image_features_extractor, mask_features_extractor):
        image_features = image_features_extractor(sample.image, shared_keys={ImageFeatures.ImageId: sample.sample_id, ImageFeatures.DatasetSplit: "N/A"})

        mask_features = mask_features_extractor(
            sample.mask, shared_keys={SegmentationMaskFeatures.ImageId: sample.sample_id, SegmentationMaskFeatures.DatasetSplit: "N/A"}
        )
        return image_features, mask_features

    @classmethod
    def extract_features(cls, dataset: SegmentationDatasetAdapter, num_workers: int) -> FeaturesCollection:
        image_features_extractor = ImageFeaturesExtractor()
        mask_features_extractor = SemanticSegmentationFeaturesExtractor(ignore_labels=dataset.get_ignored_classes())

        image_features = []
        mask_features = []

        pool_cls = Pool if num_workers > 0 else DummyPool
        with pool_cls(num_workers) as p:
            samples_iterator = tqdm(dataset.get_iterator(), total=len(dataset), desc=f"Extracting features")
            for features in p.starmap(
                cls.process_sample, zip(samples_iterator, itertools.repeat(image_features_extractor), itertools.repeat(mask_features_extractor))
            ):
                image_features.append(features[0])
                mask_features.append(features[1])

        results = FeaturesCollection(
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
    def extract_features_from_splits(self, datasets: Mapping[str, SegmentationDatasetAdapter], num_workers: int = 0) -> FeaturesCollection:
        image_features = []
        mask_features = []

        for split_name, dataset in datasets.items():
            results = self.extract_features(dataset, num_workers=num_workers)
            results.image_features[ImageFeatures.DatasetSplit] = split_name
            results.mask_features[SegmentationMaskFeatures.DatasetSplit] = split_name

            image_features.append(results.image_features)
            mask_features.append(results.mask_features)

        results = FeaturesCollection(
            image_features=pd.concat(image_features),
            mask_features=pd.concat(mask_features),
            bbox_features=None,
        )

        return results

    @classmethod
    def plot_analysis(cls, datasets: Mapping[str, SegmentationDatasetAdapter], num_workers: int = 0):
        results = cls.extract_features_from_splits(datasets, num_workers=num_workers)
        report_template = ReportTemplate.get_report_template_with_valid_widgets(results)
        NotebookWriter().write_report(results, report_template)
