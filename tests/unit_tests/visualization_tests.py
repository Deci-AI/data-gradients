import unittest

import numpy as np
import pandas as pd

from data_gradients.visualize.seaborn_renderer import SeabornRenderer, Hist2DPlotOptions, BarPlotOptions


class VisualizationTests(unittest.TestCase):
    """ """

    def setUp(self):
        train_df = pd.DataFrame.from_dict(
            {
                "image_width": np.random.normal(800, 200, size=1000),
                "image_height": np.random.normal(480, 200, size=1000),
                "split": ["train"] * 1000,
            }
        )

        valid_df = pd.DataFrame.from_dict(
            {
                "image_width": np.random.normal(400, 150, size=1000),
                "image_height": np.random.normal(980, 130, size=1000),
                "split": ["val"] * 1000,
            }
        )

        test_df = pd.DataFrame.from_dict(
            {
                "image_width": np.random.normal(800, 50, size=1000),
                "image_height": np.random.normal(480, 180, size=1000),
                "split": ["test"] * 1000,
            }
        )

        self.image_size_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

        train_df = pd.DataFrame.from_dict(
            dict(
                x=np.random.randn(1000) * 0.85 - 0.4,
                y=np.random.randn(1000) * 1.5 + 0.1,
                weight = np.random.randn(1000) * 0.5 + 0.5,
                class_name=np.random.choice(["apples", "oranges", "bananas", "kiwi"], 1000),
                split=["train"] * 1000,
            )
        )

        valid_df = pd.DataFrame.from_dict(
            dict(
                x=np.random.randn(1000) * 1.45 + 0.4,
                y=np.random.randn(1000) * 0.5 - 0.1,
                weight=np.random.randn(1000) * 0.5 + 0.5,
                class_name=np.random.choice(["liche", "apples", "oranges", "bananas"], 1000),
                split=["valid"] * 1000,
            )
        )

        self.fruits_df = pd.concat([train_df, valid_df], ignore_index=True)

    def test_plot_image_size_hist_color_by_split(self):
        options = Hist2DPlotOptions(
            figsize=(10, 10),
            title="Image size distribution",
            x_label_name="Image width (pixels)",
            x_label_key="image_width",
            y_label_name="Image height (pixels)",
            y_label_key="image_height",
            labels_key="split",
            labels_palette={"train": "royalblue", "val": "red", "test": "limegreen"},
            bins=32,
            kde=True
        )

        sns = SeabornRenderer()
        sns.render_with_options(self.image_size_df, options).show()

    def test_plot_image_size_hist_individual_plots(self):
        options = Hist2DPlotOptions(
            figsize=(15, 5),
            title="Image size distribution",
            x_label_name="Image width (pixels)",
            x_label_key="image_width",
            y_label_name="Image height (pixels)",
            y_label_key="image_height",
            labels_key="split",
            labels_palette={"train": "royalblue", "val": "red", "test": "limegreen"},
            bins=32,
            individual_plots_key="split",
            individual_plots_max_cols=3,
        )

        sns = SeabornRenderer()
        sns.render_with_options(self.image_size_df, options).show()

    def test_barplot_visualization_class_distribution(self):
        options = BarPlotOptions(
            x_label_key="class_name",
            x_label_name="Fruits",
            y_label_key=None,
            y_label_name="Count",
            title="Fruit distribution with class imbalance",
            x_ticks_rotation=None,
            labels_key="split",
            log_scale=False,
        )

        sns = SeabornRenderer()
        sns.render_with_options(self.fruits_df, options).show()

    def test_barplot_visualization_split_distribution(self):
        options = BarPlotOptions(
            x_label_key="split",
            x_label_name="Split",
            y_label_key=None,
            y_label_name="Count",
            title="Class distribution within each split",
            x_ticks_rotation=None,
            labels_key="class_name",
            log_scale=False,
        )

        sns = SeabornRenderer()
        sns.render_with_options(self.fruits_df, options).show()

    def test_barplot_visualization_weight_distribution(self):
        options = BarPlotOptions(
            x_label_key="class_name",
            x_label_name="Fruits",
            y_label_key="weight",
            y_label_name="Weight",
            title="Fruit weight distribution",
            x_ticks_rotation=None,
            labels_key="split",
            log_scale=False,
        )

        sns = SeabornRenderer()
        sns.render_with_options(self.fruits_df, options).show()


if __name__ == "__main__":
    unittest.main()
