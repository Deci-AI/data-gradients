from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from data_gradients.utils.data_classes.extractor_results import HeatMapResults, HistogramResults


def create_json_object(values, keys):
    return dict(zip(keys, list(values)))


def write_bar_plot(ax, results: HistogramResults):
    if results.ax_grid:
        ax.grid(visible=True, axis="y")

    number_of_labels = len(results.bins)
    ax.bar(
        x=np.arange(number_of_labels) - (results.width / 2 if (results.split == "train") else -results.width / 2),
        height=results.values,
        width=results.width,
        label=results.split,
        color=results.color,
    )

    plt.xticks(
        ticks=[label for label in range(number_of_labels)],
        labels=results.bins,
        rotation=results.ticks_rotation,
    )

    if results.y_ticks:
        for i in range(len(results.bins)):
            v = np.round(results.values[i], 2) if np.round(results.values[i], 2) > 0.0 else ""
            plt.text(
                x=i - (results.width / 2 if (results.split == "train") else -results.width / 2),
                y=1.01 * results.values[i],
                s=v,
                ha="center",
                size="xx-small",
            )

    ax.set_xlabel(results.x_label)
    ax.set_ylabel(results.y_label)
    ax.set_title(results.title)
    ax.legend()


def write_heatmap_plot(ax, results: HeatMapResults, fig=None):
    if results.n_bins == 0:
        # Set to default
        results.n_bins = 10

    if not results.range:
        results.range = [
            [0, min(max(results.x) + 0.1, 1)],
            [0, min(max(results.y) + 0.1, 1)],
        ]
    if results.invert:
        # Bug occurs only in center of mass feature extractor!
        # BUG - All results are inverted
        results.x = [abs(x - 1) for x in results.x]
        results.y = [abs(y - 1) for y in results.y]

    hh = ax.hist2d(
        x=results.x,
        y=results.y,
        bins=(results.n_bins, results.n_bins),
        range=results.range,
        cmap="Reds",
        vmin=0,
    )

    if fig is not None:
        fig.colorbar(hh[3], ax=ax)

    ax.set_xlabel(results.x_label)
    ax.set_ylabel(results.y_label)
    ax.set_title(results.split.capitalize() + " - " + results.title)


def class_id_to_name(mapping, hist: Dict):
    if mapping is None:
        return hist

    new_hist = {}
    for key in list(hist.keys()):
        try:
            new_hist.update({mapping[key]: hist[key]})
        except KeyError:
            new_hist.update({key: hist[key]})
    return new_hist
