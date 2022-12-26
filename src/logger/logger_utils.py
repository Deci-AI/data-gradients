import numpy
import numpy as np
from matplotlib import pyplot as plt, cm
from scipy.ndimage import gaussian_filter


def create_json_object(values, keys):
    return dict(zip(keys, list(values)))


def create_bar_plot(ax, data, labels, split: str, x_label: str = "", y_label: str = "", title: str = "",
                    width: float = 0.4, ticks_rotation: int = 270,
                    color: str = 'yellow', yticks: bool = False):

    number_of_labels = len(labels)
    ax.bar(x=np.arange(number_of_labels) - (width / 2 if (split == 'train') else -width / 2),
           height=data,
           width=width,
           label=split,
           color=color)

    plt.xticks(ticks=[label for label in range(number_of_labels)],
               labels=labels,
               rotation=ticks_rotation)

    if yticks:
        for i in range(len(labels)):
            v = np.round(data[i], 1) if np.round(data[i], 1) > 0. else ""
            plt.text(i - (width / 2 if (split == 'train') else -width / 2), data[i], v, ha='center', size='x-small')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()


def create_heatmap_plot(ax, x, y, split: str, bins=50, sigma=2, title="", x_label="", y_label="", use_gaussian_filter: bool=True, use_extent: bool=True):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    if use_gaussian_filter:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    if use_extent:
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap=cm.jet)
    else:
        ax.imshow(heatmap.T, origin='lower', aspect='auto', cmap=cm.jet)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(split.capitalize() + " - " + title)


