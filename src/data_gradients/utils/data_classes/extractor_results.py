from dataclasses import dataclass, field
from typing import List
from abc import ABC


@dataclass
class VisualizationResults(ABC):
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


@dataclass
class HeatMapResults(HistoResults):
    x: List = field(default_factory=list)
    y: List = field(default_factory=list)
    keys: List = field(default_factory=list)

    n_bins: int = 50
    range: List = field(default_factory=list)
    invert: bool = False
