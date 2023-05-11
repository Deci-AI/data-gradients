import collections
import unittest

import cv2
import numpy as np
import pandas as pd

from data_gradients.example_dataset.bdd_dataset import BDDDataset
from data_gradients.feature_extractors.features import SegmentationMaskFeatures
from data_gradients.feature_extractors.image_features_extractor import ImageFeaturesExtractor
from data_gradients.feature_extractors.result import FeaturesResult
from data_gradients.feature_extractors.segmentation.segmentation_features_extractor import \
    SemanticSegmentationFeaturesExtractor
from data_gradients.logging import MarkdownWriter, HTMLWriter, PDFWriter
from data_gradients.reports.report_template import ReportTemplate


class SemanticSegmentationFeaturesExtractorTest(unittest.TestCase):
    def test_simple_image(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        extractor = SemanticSegmentationFeaturesExtractor(ignore_labels=0)
        dict_df = extractor(image)
        df = pd.DataFrame.from_dict(dict_df)
        self.assertEquals(len(df), 0)

    def test_image_with_simple_shapes(self):
        image = np.zeros((1024, 1024), dtype=np.uint8)
        image[700:, 700:] = 255
        cv2.circle(image, (100, 100), 50, 1, -1)
        cv2.rectangle(image, (500, 500), (600, 600), 2, -1)

        extractor = SemanticSegmentationFeaturesExtractor(ignore_labels=[0, 255])
        dict_df = extractor(image)
        df = pd.DataFrame.from_dict(dict_df)
        self.assertEquals(len(df), 2)
        print(df)

    def test_image_with_simple_shapes_no_ignore(self):
        image = np.zeros((1024, 1024), dtype=np.uint8)
        image[700:, 700:] = 255
        cv2.circle(image, (100, 100), 50, 1, -1)
        cv2.rectangle(image, (500, 500), (600, 600), 2, -1)

        extractor = SemanticSegmentationFeaturesExtractor(ignore_labels=None)
        dict_df = extractor(image)
        df = pd.DataFrame.from_dict(dict_df)
        self.assertEquals(len(df), 4)
        print(df)

    def test_bdd(self):
        mask_features_extractor = SemanticSegmentationFeaturesExtractor(ignore_labels=[255])
        image_features_extractor = ImageFeaturesExtractor()

        extractors = {"image": image_features_extractor, "mask": mask_features_extractor}

        train_dataset = BDDDataset(
            data_folder="../../src/data_gradients/example_dataset/bdd_example",
            split="train",
        )
        val_dataset = BDDDataset(
            data_folder="../../src/data_gradients/example_dataset/bdd_example",
            split="val",
        )

        results = collections.defaultdict(list)

        for image_id, (image, mask_image) in enumerate(train_dataset):
            shared_keys = {SegmentationMaskFeatures.DatasetSplit: "train", SegmentationMaskFeatures.ImageId: image_id}

            # TODO: Unpacking of dataset sample to specific feature extractor should be done in a more elegant way
            for extractor_name, extractor in extractors.items():
                dict_df = extractor(mask_image if extractor_name == "mask" else image, shared_keys=shared_keys)
                df = pd.DataFrame.from_dict(dict_df)
                results[extractor_name].append(df)

        for image_id, (image, mask_image) in enumerate(val_dataset):
            shared_keys = {SegmentationMaskFeatures.DatasetSplit: "val", SegmentationMaskFeatures.ImageId: image_id}

            for extractor_name, extractor in extractors.items():
                dict_df = extractor(mask_image if extractor_name == "mask" else image, shared_keys=shared_keys)
                df = pd.DataFrame.from_dict(dict_df)
                results[extractor_name].append(df)

        results = FeaturesResult(
            image_features=pd.concat(results["image"]),
            mask_features=pd.concat(results["mask"]),
            bbox_features=None,
        )

        # This is a bit ugly - after we assembled all the features, we enrich our dataframes with additional columns
        # like class name. The class name helps with plotting to make the plots more readable (E.g. instead of 0,1,2,3 user would see car,plane,truck, etc.)
        results.mask_features[SegmentationMaskFeatures.SegmentationMaskLabelName] = results.mask_features[SegmentationMaskFeatures.SegmentationMaskLabel].apply(
            lambda x: train_dataset.CLASS_ID_TO_NAMES[x]
        )

        # This is the report definition. It is hardcoded right now, but can come from config file as well.
        # Each item represents a single report widget.
        report = ReportTemplate.get_report_template_with_valid_widgets(results)

        MarkdownWriter(output_file="markdown/bdd_report.md", images_subfolder="img").write_report(results, report)
        HTMLWriter(output_file="html/bdd_report.html", images_subfolder="img").write_report(results, report)
        PDFWriter(output_file="pdf/bdd_report.pdf").write_report(results, report)
