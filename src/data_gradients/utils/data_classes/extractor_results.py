from dataclasses import dataclass, field
from typing import List


@dataclass()
class Results:
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


@dataclass()
class HeatMapResults(Results):
    x: List = field(default_factory=list)
    y: List = field(default_factory=list)
    keys: List = field(default_factory=list)

    n_bins: int = 50
    sigma: int = 2

    use_extent: bool = True
    use_gaussian_filter: bool = True,

