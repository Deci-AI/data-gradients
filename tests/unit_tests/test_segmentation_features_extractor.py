import os
import unittest

import cv2
import numpy as np
import pandas as pd
import torchvision

from data_gradients.dataset_adapters import TorchvisionCityscapesSegmentationAdapter, BDD100KSegmentationDatasetAdapter
from data_gradients.feature_extractors import SemanticSegmentationFeaturesExtractor
from data_gradients.writers import MarkdownWriter, HTMLWriter, PDFWriter, TensorboardWriter
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager
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
        train_dataset = BDD100KSegmentationDatasetAdapter(
            data_dir="../../src/data_gradients/example_dataset/bdd_example/train",
        )
        val_dataset = BDD100KSegmentationDatasetAdapter(
            data_dir="../../src/data_gradients/example_dataset/bdd_example/val",
        )

        data = {
            "train": train_dataset,
            "val": val_dataset,
        }

        results = SegmentationAnalysisManager.extract_features_from_splits(data)

        report = ReportTemplate.get_report_template_with_valid_widgets(results)

        MarkdownWriter(output_file="bdd/report.md", images_subfolder="markdown_images").write_report(results, report)
        HTMLWriter(output_file="bdd/report.html", images_subfolder="html_images").write_report(results, report)
        PDFWriter(output_file="bdd/report.pdf").write_report(results, report)
        TensorboardWriter(output_directory="bdd/tensorboard").write_report(results, report)
