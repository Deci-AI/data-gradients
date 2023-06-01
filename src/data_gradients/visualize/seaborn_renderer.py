import numpy as np
import pandas as pd
import seaborn
from typing import Union, List, Dict
from matplotlib import pyplot as plt

from data_gradients.visualize.plot_options import (
    PlotRenderer,
    CommonPlotOptions,
    Hist2DPlotOptions,
    BarPlotOptions,
    ScatterPlotOptions,
    ViolinPlotOptions,
    KDEPlotOptions,
    ImagePlotOptions,
)

__all__ = ["SeabornRenderer"]


class SeabornRenderer(PlotRenderer):
    def __init__(self, style="whitegrid", palette="pastel"):
        seaborn.set_theme(style=style, palette=palette)

    def render(self, data: Union[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]], options: CommonPlotOptions) -> List[plt.Figure]:
        """Plot a graph using seaborn.

        :param data:    The data to render. It has to include the fields listed in the options.
        :param options: The plotting options, which includes the information about the type of plot and the parameters required to plot it.
        :return:        The matplotlib figure.
        """
        if isinstance(options, Hist2DPlotOptions):
            return self._render_histplot(data, options)
        if isinstance(options, BarPlotOptions):
            return self._render_barplot(data, options)
        if isinstance(options, ScatterPlotOptions):
            return self._render_scatterplot(data, options)
        if isinstance(options, ViolinPlotOptions):
            return self._render_violinplot(data, options)
        if isinstance(options, KDEPlotOptions):
            return self._render_kdeplot(data, options)
        if isinstance(options, ImagePlotOptions):
            return self._render_images(data, options)

        raise ValueError(f"Unknown options type: {type(options)}")

    def _render_scatterplot(self, df, options: ScatterPlotOptions) -> List[plt.Figure]:

        if options.individual_plots_key is None:
            dfs = [df]
            n_rows = 1
            n_cols = 1
        else:
            dfs = [df[df[options.individual_plots_key] == key] for key in df[options.individual_plots_key].unique()]
            _num_images = len(dfs)
            _max_cols = options.individual_plots_max_cols
            n_cols = _num_images if _max_cols is None else min(_num_images, _max_cols)
            n_rows = int(np.ceil(_num_images / n_cols))

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
            scatterplot_args = dict(
                data=df,
                x=options.x_label_key,
                y=options.y_label_key,
                ax=ax_i,
            )

            if options.labels_key is not None:
                scatterplot_args.update(hue=options.labels_key)
                if options.labels_palette is not None:
                    scatterplot_args.update(palette=options.labels_palette)

            seaborn.scatterplot(**scatterplot_args)

            if options.x_lim is not None:
                ax_i.set_xlim(options.x_lim)

            if options.y_lim is not None:
                ax_i.set_ylim(options.y_lim)

            ax_i.set_xlabel(options.x_label_name)
            ax_i.set_ylabel(options.y_label_name)
            if options.labels_name is not None:
                ax_i.legend(title=options.labels_name)

            if options.x_ticks_rotation == "auto":
                n_unique = len(df[options.x_label_key].unique())
                if n_unique > 50:
                    options.x_ticks_rotation = 90
                elif n_unique > 10:
                    options.x_ticks_rotation = 45

            self._set_ticks_rotation(ax_i, options.x_ticks_rotation, options.y_ticks_rotation)

        return [fig]

    def _render_histplot(self, df, options: Hist2DPlotOptions) -> List[plt.Figure]:

        if options.individual_plots_key is None:
            dfs = [df]
            n_rows = 1
            n_cols = 1
        else:
            dfs = [df[df[options.individual_plots_key] == key] for key in df[options.individual_plots_key].unique()]
            _num_images = len(dfs)
            _max_cols = options.individual_plots_max_cols
            n_cols = _num_images if _max_cols is None else min(_num_images, _max_cols)
            n_rows = int(np.ceil(_num_images / n_cols))

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
            histplot_args = dict(data=df, x=options.x_label_key, kde=options.kde, stat=options.stat, ax=ax_i)

            if options.y_label_key is not None:
                histplot_args.update(y=options.y_label_key)

            if options.weights is not None:
                histplot_args.update(weights=options.weights)

            if options.y_label_key is not None:
                histplot_args.update(y=options.y_label_key)

            if options.weights is not None:
                histplot_args.update(weights=options.weights)

            if options.bins is not None:
                histplot_args.update(bins=options.bins)

            if options.labels_key is not None:
                histplot_args.update(hue=options.labels_key)
                if options.labels_palette is not None:
                    histplot_args.update(palette=options.labels_palette)

            seaborn.histplot(**histplot_args)

            ax_i.set_xlabel(options.x_label_name)
            if options.y_label_name is not None:
                ax_i.set_ylabel(options.y_label_name)

            if options.labels_name is not None:
                ax_i.legend(title=options.labels_name)

            if options.x_lim is not None:
                ax_i.set_xlim(options.x_lim)

            if options.y_lim is not None:
                ax_i.set_ylim(options.y_lim)

            if options.x_ticks_rotation == "auto":
                n_unique = len(df[options.x_label_key].unique())
                if n_unique > 50:
                    options.x_ticks_rotation = 90
                elif n_unique > 10:
                    options.x_ticks_rotation = 45

            self._set_ticks_rotation(ax_i, options.x_ticks_rotation, options.y_ticks_rotation)

        return fig

    def _render_kdeplot(self, df, options: KDEPlotOptions) -> List[plt.Figure]:

        if options.individual_plots_key is None:
            dfs = [df]
            n_rows = 1
            n_cols = 1
        else:
            dfs = [df[df[options.individual_plots_key] == key] for key in df[options.individual_plots_key].unique()]
            _num_images = len(dfs)
            _max_cols = options.individual_plots_max_cols
            n_cols = _num_images if _max_cols is None else min(_num_images, _max_cols)
            n_rows = int(np.ceil(_num_images / n_cols))

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
            plot_args = dict(
                data=df,
                x=options.x_label_key,
                ax=ax_i,
                common_norm=options.common_norm,
            )

            if options.fill:
                plot_args.update(fill=options.fill, alpha=options.alpha)

            if options.bw_adjust is not None:
                plot_args.update(bw_adjust=options.bw_adjust)

            if options.y_label_key is not None:
                plot_args.update(y=options.y_label_key)

            if options.weights is not None:
                plot_args.update(weights=options.weights)

            if options.labels_key is not None:
                plot_args.update(hue=options.labels_key)
                if options.labels_palette is not None:
                    plot_args.update(palette=options.labels_palette)

            seaborn.kdeplot(**plot_args)

            ax_i.set_xlabel(options.x_label_name)
            if options.y_label_name is not None:
                ax_i.set_ylabel(options.y_label_name)

            if options.labels_name is not None:
                ax_i.legend(title=options.labels_name)

            if options.x_lim is not None:
                ax_i.set_xlim(options.x_lim)

            if options.y_lim is not None:
                ax_i.set_ylim(options.y_lim)

            if options.x_ticks_rotation == "auto":
                n_unique = len(df[options.x_label_key].unique())
                if n_unique > 50:
                    options.x_ticks_rotation = 90
                elif n_unique > 10:
                    options.x_ticks_rotation = 45

            self._set_ticks_rotation(ax_i, options.x_ticks_rotation, options.y_ticks_rotation)

        return [fig]

    def _render_violinplot(self, df, options: ViolinPlotOptions) -> List[plt.Figure]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=options.figsize)
        if options.tight_layout:
            fig.tight_layout()
        fig.suptitle(options.title)
        fig.subplots_adjust(top=0.9)

        plot_args = dict(
            data=df,
            x=options.x_label_key,
            y=options.y_label_key,
            ax=ax,
        )

        if options.order_key is not None:
            if options.order_key not in df.columns:
                raise ValueError(f"{options.order_key} is not a column in {df.columns}")
            sorted_df = df.sort_values(options.order_key)
            sorted_labels = sorted_df[options.y_label_key].unique()
            plot_args.update(order=sorted_labels)

        if options.bandwidth is not None:
            plot_args.update(bw=options.bandwidth)

        if options.labels_key is not None:
            plot_args.update(hue=options.labels_key, split=True)
            if options.labels_palette is not None:
                plot_args.update(palette=options.labels_palette)

        ax = seaborn.violinplot(**plot_args)

        ax.set_xlabel(options.x_label_name)
        ax.set_ylabel(options.y_label_name)

        if options.x_lim is not None:
            ax.set_xlim(options.x_lim)

        if options.labels_name is not None:
            ax.legend(title=options.labels_name)

        if options.x_ticks_rotation == "auto":
            n_unique = len(df[options.x_label_key].unique())
            if n_unique > 50:
                options.x_ticks_rotation = 90
            elif n_unique > 10:
                options.x_ticks_rotation = 45

        self._set_ticks_rotation(ax, options.x_ticks_rotation, options.y_ticks_rotation)

        return [fig]

    def _render_barplot(self, df, options: BarPlotOptions) -> List[plt.Figure]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=options.figsize)
        if options.tight_layout:
            fig.tight_layout()
        fig.suptitle(options.title)
        fig.subplots_adjust(top=0.9)

        barplot_args = dict(
            data=df,
            x=options.x_label_key,
            width=options.width,
            ax=ax,
            orient=options.orient,
        )

        if options.y_label_key is not None:
            barplot_args.update(y=options.y_label_key)
            plot_fn = seaborn.barplot
        else:
            plot_fn = seaborn.countplot

        if options.order_key is not None:
            if options.order_key not in df.columns:
                raise ValueError(f"{options.order_key} is not a column in {df.columns}")
            sorted_df = df.sort_values(options.order_key)
            sorted_labels = sorted_df[options.y_label_key].unique()
            barplot_args.update(order=sorted_labels)

        if options.bins is not None:
            barplot_args.update(bins=options.bins)

        if options.labels_key is not None:
            barplot_args.update(hue=options.labels_key)
            if options.labels_palette is not None:
                barplot_args.update(palette=options.labels_palette)

        ax = plot_fn(**barplot_args)

        ax.set_xlabel(options.x_label_name)
        ax.set_ylabel(options.y_label_name)
        if options.labels_name is not None:
            ax.legend(title=options.labels_name)

        if options.log_scale is True:
            ax.set_yscale("log")
            ax.set_ylabel(options.y_label_name + " (log scale)")

        if options.x_ticks_rotation == "auto":
            n_unique = len(df[options.x_label_key].unique())
            if n_unique > 50:
                options.x_ticks_rotation = 90
            elif n_unique > 10:
                options.x_ticks_rotation = 45

        if options.show_values:
            self._show_values(ax)

        self._set_ticks_rotation(ax, options.x_ticks_rotation, options.y_ticks_rotation)

        return [fig]

    def _render_images(self, images_per_split_per_class: Dict[str, Dict[str, np.ndarray]], options: ImagePlotOptions) -> List[plt.Figure]:
        """Render images using matplotlib. Plot one graph with all splits per class.

        :param images_per_split_per_class:  Mapping of class names and splits to images. e.g. {"class1": {"train": np.ndarray, "valid": np.ndarray},...}
        :param options:                     Plotting options
        """
        n_class_per_row = 2

        figs = []

        for i, (class_name, images_per_split) in enumerate(images_per_split_per_class.items()):
            n_cols = len(images_per_split)
            fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, 6))

            if options.tight_layout:
                fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            fig.suptitle(f"Class: {class_name}", fontsize=36)

            for (split, image), ax_i in zip(images_per_split.items(), axs):
                ax_i.imshow(image, cmap="hot")

                if i >= n_class_per_row:
                    ax_i.set_axis_off()
                else:
                    # Write the split for the last row only
                    ax_i.set_xticks([])
                    ax_i.set_yticks([])
                    ax_i.spines["top"].set_visible(False)
                    ax_i.spines["right"].set_visible(False)
                    ax_i.spines["bottom"].set_visible(False)
                    ax_i.spines["left"].set_visible(False)
                    ax_i.set_title(split, fontsize=48)
                    # ax_i.xaxis.set_label_coords(0.5, -0.05)  # Move a little down to increase readability

            figs.append(fig)

        import io
        from PIL import Image

        # Convert figs to PIL Images
        images = []
        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            images.append(Image.open(buf))
            plt.close(fig)  # Close the figure

        # Create a new figure to hold the images
        n_rows = len(images) // n_class_per_row + len(images) % n_class_per_row
        n_cols = n_class_per_row

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 2.5 * n_rows))  # Reduced the vertical size

        # Plot each PIL image to a subplot
        from itertools import zip_longest

        for ax, img in zip_longest(axs.flatten(), images, fillvalue=None):
            ax.axis("off")
            if img is not None:
                ax.imshow(img)

        plt.tight_layout()
        return [fig]

    def _set_ticks_rotation(self, ax, x_ticks_rotation, y_ticks_rotation):
        # Call to set_xticks is needed to avoid warning
        # https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator

        if x_ticks_rotation is not None:
            ax.set_xticks(list(ax.get_xticks()))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=x_ticks_rotation)

        if y_ticks_rotation is not None:
            ax.set_yticks(list(ax.get_yticks()))
            ax.set_yticklabels(ax.get_yticklabels(), rotation=y_ticks_rotation)

    def _show_values(self, axs, orient="v", space=0.01):
        def _single(ax):
            if orient == "v":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height() + (p.get_height() * 0.01)
                    value = "{:.1f}".format(p.get_height())
                    ax.text(_x, _y, value, ha="center")
            elif orient == "h":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() + float(space)
                    _y = p.get_y() + p.get_height() - (p.get_height() * 0.5)
                    value = "{:.1f}".format(p.get_width())
                    ax.text(_x, _y, value, ha="left")

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _single(ax)
        else:
            _single(axs)
