from functools import partial
from multiprocessing import Pool
from typing import Mapping, Union, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_gradients.dataset_adapters import SegmentationDatasetAdapter
from data_gradients.dataset_adapters.adapter_interface import ImageSample
from data_gradients.feature_extractors import (
    FeaturesCollection,
    ImageFeatures,
    ImageFeaturesExtractor,
)
from data_gradients.managers.abstract_manager import AnalysisManagerAbstract
from data_gradients.managers.utils.dummy_pool import DummyPool
from data_gradients.reports.report_template import ReportTemplate
from data_gradients.writers import NotebookWriter


class ImageAnalysisManager(AnalysisManagerAbstract):
    @classmethod
    def extract_features(cls, dataset: SegmentationDatasetAdapter, num_workers: int, max_samples: Optional[int] = None) -> FeaturesCollection:
        image_features_extractor = ImageFeaturesExtractor()

        image_features = []

        # maxtasksperchild=1 is necessary to avoid growing memory usage when using multiprocessing
        # I'm not sure
        pool_cls = partial(Pool, maxtasksperchild=1) if num_workers > 0 else DummyPool

        _process_sample_fn = partial(
            cls.process_sample,
            dataset=dataset,
            image_features_extractor=image_features_extractor,
        )

        indexes = np.arange(len(dataset))
        if max_samples is not None:
            indexes = indexes[:max_samples]

        with pool_cls(num_workers) as p:
            for features in tqdm(
                p.imap_unordered(_process_sample_fn, indexes, chunksize=1),
                total=len(indexes),
                desc=f"Extracting features",
            ):
                image_features.append(features)

        results = FeaturesCollection(
            image_features=pd.concat(image_features),
            mask_features=None,
            bbox_features=None,
        )

        return results

    @classmethod
    def process_sample(cls, index: int, dataset, image_features_extractor):
        sample: ImageSample = dataset[index]
        image_features = image_features_extractor(sample.image, shared_keys={ImageFeatures.ImageId: sample.sample_id, ImageFeatures.DatasetSplit: "N/A"})

        return pd.DataFrame.from_dict(image_features)

    @classmethod
    def extract_features_from_splits(
        self, datasets: Mapping[str, SegmentationDatasetAdapter], num_workers: int = 0, max_samples: Optional[int] = None
    ) -> FeaturesCollection:
        image_features = []

        for split_name, dataset in datasets.items():
            results = self.extract_features(dataset, num_workers=num_workers, max_samples=max_samples)
            results.image_features[ImageFeatures.DatasetSplit] = split_name

            image_features.append(results.image_features)

        results = FeaturesCollection(
            image_features=pd.concat(image_features),
            mask_features=None,
            bbox_features=None,
        )

        return results

    @classmethod
    def plot_analysis(cls, datasets: Mapping[str, Union[SegmentationDatasetAdapter, DataLoader]], num_workers: int = 0):
        """
        Extracts features from the dataset and plots them.
        """

        results = cls.extract_features_from_splits(datasets, num_workers=num_workers)
        report_template = ReportTemplate.get_report_template_with_valid_widgets(results)
        NotebookWriter().write_report(results, report_template)
