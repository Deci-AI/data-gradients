from typing import Dict

import numpy as np
from matplotlib import pyplot as plt, cm
from scipy.ndimage import gaussian_filter

from data_gradients.utils.data_classes.extractor_results import HeatMapResults, Results


def create_json_object(values, keys):
    return dict(zip(keys, list(values)))


def write_bar_ploit(ax, results: Results):
    number_of_labels = len(Resultsbins)
    ax.bar(x=np.arange(number_of_labels) - (Resultswidth / 2 if (Resultssplit == 'train') else - Resultswidth / 2),
           height=Resultsvalues,
           width=Resultswidth,
           label=Resultssplit,
           color=Resultscolor)

    plt.xticks(ticks=[label for label in range(number_of_labels)],
               labels=Resultsbins,
               rotation=Resultsticks_rotation)

    if Resultsy_ticks:
        for i in range(len(Resultsbins)):
            v = np.round(Resultsvalues[i], 2) if np.round(Resultsvalues[i], 2) > 0. else ""
            plt.text(x=i - (Resultswidth / 2 if (Resultssplit == 'train') else -Resultswidth / 2),
                     y=1.01 * Resultsvalues[i],
                     s=v,
                     ha='center',
                     size='xx-small')

    ax.set_xlabel(Resultsx_label)
    ax.set_ylabel(Resultsy_label)
    ax.set_title(Resultstitle)
    ax.legend()


def write_heatmap_plot(ax, results: HeatMapResults):
    if Resultsn_bins == 0:
        Resultsn_bins = 1

    heatmap, xedges, yedges = np.histogram2d(Resultsx, Resultsy, bins=Resultsn_bins)

    if Resultsuse_gaussian_filter:
        heatmap = gaussian_filter(heatmap, sigma=Resultssigma)
    if Resultsuse_extent:
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap=cm.jet)
    else:
        ax.imshow(heatmap.T, origin='lower', aspect='auto', cmap=cm.jet)

    ax.set_xlabel(Resultsx_label)
    ax.set_ylabel(Resultsy_label)
    ax.set_title(Resultssplit.capitalize() + " - " + Resultstitle)


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


