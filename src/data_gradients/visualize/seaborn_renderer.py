import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt

__all__ = ["SeabornRenderer"]

from data_gradients.visualize.plot_options import PlotRenderer, CommonPlotOptions, Hist2DPlotOptions, BarPlotOptions, ScatterPlotOptions


class SeabornRenderer(PlotRenderer):
    def __init__(self, style="whitegrid", palette="pastel"):
        seaborn.set_theme(style=style, palette=palette)

    def render_with_options(self, df: pd.DataFrame, options: CommonPlotOptions) -> plt.Figure:
        if isinstance(options, Hist2DPlotOptions):
            return self.render_histplot(df, options)
        if isinstance(options, BarPlotOptions):
            return self.render_barplot(df, options)
        if isinstance(options, ScatterPlotOptions):
            return self.render_scatterplot(df, options)

        raise ValueError(f"Unknown options type: {type(options)}")

    def render_scatterplot(self, df, options: ScatterPlotOptions) -> plt.Figure:
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

        return fig

    def render_histplot(self, df, options: Hist2DPlotOptions) -> plt.Figure:
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
            if options.labels_name is not None:
                ax_i.legend(title=options.labels_name)

            if options.x_ticks_rotation == "auto":
                n_unique = len(df[options.x_label_key].unique())
                if n_unique > 50:
                    options.x_ticks_rotation = 90
                elif n_unique > 10:
                    options.x_ticks_rotation = 45

            self._set_ticks_rotation(ax_i, options.x_ticks_rotation, options.y_ticks_rotation)

        return fig

    def render_barplot(self, df, options: BarPlotOptions) -> plt.Figure:
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

        return fig

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
