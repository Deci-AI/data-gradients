import collections
import unittest

import cv2
import numpy as np
import pandas as pd

from data_gradients.example_dataset.bdd_dataset import BDDDataset
from data_gradients.feature_extractors.features import SegmentationMaskFeatures
from data_gradients.feature_extractors.segmentation.segmentation_features_extractor import \
    SemanticSegmentationFeaturesExtractor
from data_gradients.visualize.seaborn_renderer import BarPlotOptions, SeabornRenderer, ScatterPlotOptions


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
        extractor = SemanticSegmentationFeaturesExtractor(ignore_labels=[255])

        train_dataset = BDDDataset(
            data_folder="../../src/data_gradients/example_dataset/bdd_example",
            split="train",
        )
        val_dataset = BDDDataset(
            data_folder="../../src/data_gradients/example_dataset/bdd_example",
            split="val",
        )

        train_df = []
        for image_id, (image, mask_image) in enumerate(train_dataset):
            shared_keys = {
                SegmentationMaskFeatures.DatasetSplit: "train",
                SegmentationMaskFeatures.ImageId: image_id
            }
            dict_df = extractor(mask_image, shared_keys=shared_keys)
            df = pd.DataFrame.from_dict(dict_df)
            train_df.append(df)

        valid_df = []
        for image_id, (image, mask_image) in enumerate(val_dataset):
            shared_keys = {
                SegmentationMaskFeatures.DatasetSplit: "val",
                SegmentationMaskFeatures.ImageId: image_id
            }
            dict_df = extractor(mask_image, shared_keys=shared_keys)
            df = pd.DataFrame.from_dict(dict_df)
            valid_df.append(df)

        df = pd.concat(train_df + valid_df)
        df[SegmentationMaskFeatures.SegmentationMaskLabelName] = df[SegmentationMaskFeatures.SegmentationMaskLabel].apply(lambda x: train_dataset.CLASS_ID_TO_NAMES[x])

        report = collections.OrderedDict([
            ("bdd_class_distribution", BarPlotOptions(
                x_label_key=SegmentationMaskFeatures.SegmentationMaskLabelName,
                x_label_name="Class",
                y_label_key=None,
                y_label_name="Count",
                title="Class distribution on BDD dataset",
                x_ticks_rotation=90,
                labels_key=SegmentationMaskFeatures.DatasetSplit,
                labels_name="Split",
                log_scale=False,
            )),
            ("bdd_class_area_distribution", BarPlotOptions(
                x_label_key=SegmentationMaskFeatures.SegmentationMaskLabel,
                x_label_name="Class",
                y_label_key=SegmentationMaskFeatures.SegmentationMaskArea,
                y_label_name="Area",
                title="Class / Area on BDD dataset",
                x_ticks_rotation=None,
                labels_key=SegmentationMaskFeatures.DatasetSplit,
                labels_name="Split",
                log_scale=False,
            )),
            ("bdd_center_of_mass", ScatterPlotOptions(
                x_label_key=SegmentationMaskFeatures.SegmentationMaskCenterOfMassX,
                x_label_name="Center of mass X",
                y_label_key=SegmentationMaskFeatures.SegmentationMaskCenterOfMassY,
                y_label_name="Center of mass Y",
                title="Class label",
                labels_key=SegmentationMaskFeatures.SegmentationMaskLabelName,
                labels_name="Class",
            )),
            ("bdd_solidity_sparsity", ScatterPlotOptions(
                x_label_key=SegmentationMaskFeatures.SegmentationMaskSolidity,
                x_label_name="Solidity",
                y_label_key=SegmentationMaskFeatures.SegmentationMaskSparseness,
                y_label_name="Sparseness",
                title="Solidity / Sparseness",
                labels_key=SegmentationMaskFeatures.SegmentationMaskLabelName,
                labels_name="Class",
            ))
        ])

        sns = SeabornRenderer()

        with open("bdd_report.md", "w") as f:
            f.write("# BDD Report\n")

            for report_id, plot_options in report.items():
                plot = sns.render_with_options(df, plot_options)
                plot.savefig(f"{report_id}.png")
                plot.show()

                f.write("\n\n")
                f.write(f"![{plot_options.title}]({report_id}.png)")
                f.write("\n\n")
