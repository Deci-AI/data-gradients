from typing import Dict

import numpy as np
from matplotlib import pyplot as plt, cm
from scipy.ndimage import gaussian_filter

from data_gradients.utils.data_classes.extractor_results import HeatMapResults, Results


def create_json_object(values, keys):
    return dict(zip(keys, list(values)))


def write_bar_plot(ax, results: Results):
    if results.ax_grid:
        ax.grid(visible=True, axis='y')

    number_of_labels = len(results.bins)
    ax.bar(x=np.arange(number_of_labels) - (results.width / 2 if (results.split == 'train') else - results.width / 2),
           height=results.values,
           width=results.width,
           label=results.split,
           color=results.color)

    plt.xticks(ticks=[label for label in range(number_of_labels)],
               labels=results.bins,
               rotation=results.ticks_rotation)

    if results.y_ticks:
        for i in range(len(results.bins)):
            v = np.round(results.values[i], 2) if np.round(results.values[i], 2) > 0. else ""
            plt.text(x=i - (results.width / 2 if (results.split == 'train') else -results.width / 2),
                     y=1.01 * results.values[i],
                     s=v,
                     ha='center',
                     size='xx-small')

    ax.set_xlabel(results.x_label)
    ax.set_ylabel(results.y_label)
    ax.set_title(results.title)
    ax.legend()


def write_heatmap_plot(ax, results: HeatMapResults):
    if results.n_bins == 0:
        results.n_bins = 1

    heatmap, xedges, yedges = np.histogram2d(results.x, results.y, bins=results.n_bins)

    if results.use_gaussian_filter:
        heatmap = gaussian_filter(heatmap, sigma=results.sigma)
    if results.use_extent:
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap=cm.jet)
    else:
        ax.imshow(heatmap.T, origin='lower', aspect='auto', cmap=cm.jet)

    ax.set_xlabel(results.x_label)
    ax.set_ylabel(results.y_label)
    ax.set_title(results.split.capitalize() + " - " + results.title)


def class_id_to_name(id_to_name, hist: Dict):
    if id_to_name is None:
        return hist

    new_hist = {}
    for key in list(hist.keys()):
        try:
            new_hist.update({id_to_name[key]: hist[key]})
        except KeyError as e:
            new_hist.update({key: hist[key]})
    return new_hist


