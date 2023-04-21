from dataclasses import dataclass, field
from typing import List
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class VisualizationResults(ABC):
    @abstractmethod
    def write_plot(self, ax, fig):
        pass


@dataclass
class HistoResults(VisualizationResults):
    bins: List = field(default_factory=list)
    values: List = field(default_factory=list)

    json_values: List = field(default_factory=list)

    plot: str = ""
    split: str = ""
    title: str = ""
    color: str = "yellow"
    y_label: str = ""
    x_label: str = ""
    y_ticks: bool = False
    ax_grid: bool = False

    width: float = 0.4

    ticks_rotation: int = 45

    def write_plot(self, ax, fig):
        write_bar_plot(ax, self)


@dataclass
class HeatMapResults(HistoResults):
    x: List = field(default_factory=list)
    y: List = field(default_factory=list)
    keys: List = field(default_factory=list)

    n_bins: int = 50
    range: List = field(default_factory=list)
    invert: bool = False

    def write_plot(self, ax, fig):
        write_heatmap_plot(ax=ax[int(self.split != "train")], results=self, fig=fig)


def write_bar_plot(ax, results: HistoResults):
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
