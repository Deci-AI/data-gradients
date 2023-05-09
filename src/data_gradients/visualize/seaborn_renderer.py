import dataclasses
from typing import Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt

__all__ = ["CommonPlotOptions", "BarPlotOptions", "Hist2DPlotOptions", "SeabornRenderer"]


@dataclasses.dataclass
class CommonPlotOptions:
    title: str


@dataclasses.dataclass
class BarPlotOptions(CommonPlotOptions):
    """
    Contains a set of options for displaying a bar plot
    """

    x_label_key: str  # A key for x-axis values
    x_label_name: str  # A title for x-axis
    y_label_key: Optional[str]  # An optional key for y-axis (If None, bar plot will use count of x-axis values)
    y_label_name: str  # A title for y-axis

    width: float = 0.8
    bins: Optional[int] = None  # Generic bin parameter that can be the name of a reference rule, the number of bins, or the breaks of the bins.

    x_ticks_rotation: Optional[int] = 45  # X-ticks rotation (Helps to make more compact plots)
    y_ticks_rotation: Optional[int] = 0  # Y-ticks rotation

    labels_key: Optional[str] = None  # If you want to display multiple classes on same plot use this property to indicate column
    labels_palette: Optional[
        Mapping
    ] = None  # Setting this allows you to control the colors of the bars of each label: { "train": "royalblue", "val": "red", "test": "limegreen" }

    log_scale: Union[bool, str] = "auto"  # If True, y-axis will be displayed in log scale
    tight_layout: bool = False  # If True enables more compact layout of the plot
    figsize: Optional[Tuple[int, int]] = (10, 6)  # Size of the figure


@dataclasses.dataclass
class Hist2DPlotOptions(CommonPlotOptions):
    """
    Contains a set of options for displaying a bivariative histogram plot
    """

    x_label_key: str  # A key for x-axis values
    x_label_name: str  # A title for x-axis

    y_label_key: str  # An optional key for y-axis
    y_label_name: str  # A title for y-axis

    bins: Optional[int] = None  # Generic bin parameter that can be the name of a reference rule, the number of bins, or the breaks of the bins.
    kde: bool = False

    individual_plots_key: str = None  # If not None, will create a separate plot for each unique value of this column
    individual_plots_max_cols: int = None  # Sets the maximum number of columns to plot in the individual plots

    labels_key: Optional[str] = None  # If you want to display multiple classes on same plot use this property to indicate column
    labels_palette: Optional[
        Mapping
    ] = None  # Setting this allows you to control the colors of the bars of each label: { "train": "royalblue", "val": "red", "test": "limegreen" }

    tight_layout: bool = False
    figsize: Optional[Tuple[int, int]] = (10, 6)


class SeabornRenderer:
    def __init__(self, style="whitegrid", palette="pastel"):
        seaborn.set_theme(style=style, palette=palette)

    def render_with_options(self, df: pd.DataFrame, options):
        if isinstance(options, Hist2DPlotOptions):
            return self.render_histplot(df, options)
        if isinstance(options, BarPlotOptions):
            return self.render_barplot(df, options)

    def render_histplot(self, df, options: Hist2DPlotOptions):
        dfs = []

        if options.individual_plots_key is not None:
            for key in df[options.individual_plots_key].unique():
                df_key = df[df[options.individual_plots_key] == key]
                dfs.append(df_key)
        else:
            dfs.append(df)

        if len(dfs) == 1:
            n_rows = 1
            n_cols = 1
        else:
            num_images = len(dfs)
            max_cols = options.individual_plots_max_cols
            n_cols = min(num_images, max_cols)
            n_rows = int(np.ceil(num_images / n_cols))

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=options.figsize)
        if options.tight_layout:
            fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.suptitle(options.title)

        if n_rows == 1 and n_cols == 1:
            axs = [axs]
        else:
            axs = axs.reshape(-1)

        for df, ax_i in zip(dfs, axs):
            histplot_args = dict(
                data=df,
                x=options.x_label_key,
                y=options.y_label_key,
                kde=options.kde,
                ax=ax_i,
            )

            if options.bins is not None:
                histplot_args.update(bins=options.bins)

            if options.labels_key is not None:
                histplot_args.update(hue=options.labels_key)
                if options.labels_palette is not None:
                    histplot_args.update(palette=options.labels_palette)

            seaborn.histplot(**histplot_args)

            ax_i.set_xlabel(options.x_label_name)
            ax_i.set_ylabel(options.y_label_name)
            # ax.set_title()

        return fig

    def render_barplot(self, df, options: BarPlotOptions):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=options.figsize)
        if options.tight_layout:
            fig.tight_layout()

        barplot_args = dict(
            data=df,
            x=options.x_label_key,
            width=options.width,
            ax=ax,
        )

        if options.y_label_key is not None:
            barplot_args.update(y=options.y_label_key)
            plot_fn = seaborn.barplot
        else:
            plot_fn = seaborn.countplot

        if options.bins is not None:
            barplot_args.update(bins=options.bins)

        if options.labels_key is not None:
            barplot_args.update(hue=options.labels_key)
            if options.labels_palette is not None:
                barplot_args.update(palette=options.labels_palette)

        ax = plot_fn(**barplot_args)
        ax.set_title(options.title)

        if options.x_ticks_rotation is not None:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=options.x_ticks_rotation)

        if options.y_ticks_rotation is not None:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=options.y_ticks_rotation)

        # else:
        #     n_unique = len(items[options.x_label].unique())
        #     if n_unique > 50:
        #         ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        #     elif n_unique > 10:
        #         ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        ax.set_xlabel(options.x_label_name)
        ax.set_ylabel(options.y_label_name)
        ax.set_title(options.title)

        if options.log_scale:
            ax.set_yscale("log")
            ax.set_ylabel(options.y_label_name + " (log scale)")
        # elif options.log_scale == "auto":
        #     all_values = list(itertools.chain(*[hist.bin_values for hist in data.values()]))
        #     if len(all_values):
        #         min_value = np.min(all_values)
        #         max_value = np.max(all_values)
        #         if np.log10(max_value - min_value + 1) > 3:
        #             ax.set_yscale("log")
        #             ax.set_ylabel(options.y_label_name + " (log scale)")

        return fig
