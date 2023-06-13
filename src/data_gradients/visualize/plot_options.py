from abc import ABC, abstractmethod
import dataclasses
from typing import Mapping, Optional, Tuple, Union, List

import pandas as pd


@dataclasses.dataclass
class CommonPlotOptions(ABC):
    title: str


@dataclasses.dataclass
class BarPlotOptions(CommonPlotOptions):
    """
    Contains a set of options for displaying a bar plot

    :attr x_label_key: A key for x-axis values
    :attr x_label_name: A title for x-axis
    :attr y_label_key: An optional key for y-axis (If None, bar plot will use count of x-axis values)
    :attr y_label_name: A title for y-axis
    :attr order_key: Key that will be used to order the violins. If None, the order will be automatically determined.
    :attr width: Width of the bars
    :attr bins: Generic bin parameter that can be the name of a reference rule, the number of bins, or the breaks of the bins.
    :attr x_ticks_rotation: X-ticks rotation (Helps to make more compact plots)
    :attr y_ticks_rotation: Y-ticks rotation
    :attr labels_key: If you want to display multiple classes on same plot use this property to indicate column
    :attr labels_palette: Setting this allows you to control the colors of the bars of each label: { "train": "royalblue", "val": "red", "test": "limegreen" }
    :attr log_scale: If True, y-axis will be displayed in log scale
    :attr tight_layout: If True enables more compact layout of the plot
    :attr figsize: Size of the figure
    :attr show_values: If True, will display the values of the bars above them
    """

    x_label_key: str
    x_label_name: str
    y_label_key: Optional[str]
    y_label_name: str

    order_key: Optional[str] = None

    width: float = 0.8
    bins: Optional[int] = None

    x_ticks_rotation: Optional[int] = 45
    y_ticks_rotation: Optional[int] = None

    labels_key: Optional[str] = None
    labels_name: Optional[str] = None
    labels_palette: Optional[Mapping] = None

    show_values: bool = False

    orient: str = "h"
    log_scale: Union[bool, str] = "auto"
    tight_layout: bool = False
    figsize: Optional[Tuple[int, int]] = (10, 6)


@dataclasses.dataclass
class ViolinPlotOptions(CommonPlotOptions):
    """
    Contains a set of options for displaying a violin distribution plot.

    :attr x_label_key: A key for x-axis values
    :attr x_label_name: A title for x-axis
    :attr y_label_key: An optional key for y-axis (If None, bar plot will use count of x-axis values)
    :attr y_label_name: A title for y-axis
    :attr order_key: Key that will be used to order the violins. If None, the order will be automatically determined.
    :attr x_lim: X-axis limits
    :attr bins: Generic bin parameter that can be the name of a reference rule, the number of bins, or the breaks of the bins.
    :attr kde: If True, will display a kernel density estimate
    :attr individual_plots_key: If None, the data will be displayed in a single plot.
                                If not None, will create a separate plot for each unique value of this column.
                                    e.g. `individual_plots_key="class_id"` will create a separate violin plot for each class.
    :attr individual_plots_max_cols: Sets the maximum number of columns to plot in the individual plots
    :attr labels_key: If you want to display multiple classes on same plot use this property to indicate column
    :attr bandwidth: If None, use the default bandwidth of the violin plot. Affects the kernel estimation.
    :attr labels_palette: Setting this allows you to control the colors of the bars of each label: { "train": "royalblue", "val": "red", "test": "limegreen" }
    :attr tight_layout: If True enables more compact layout of the plot
    :attr figsize: Size of the figure
    """

    x_label_key: str
    x_label_name: str

    y_label_key: str
    y_label_name: str

    order_key: Optional[str] = None

    x_lim: Tuple[float, float] = None

    individual_plots_key: str = None
    individual_plots_max_cols: int = None

    labels_key: Optional[str] = None
    labels_name: Optional[str] = None
    labels_palette: Optional[Mapping] = None

    bandwidth: Union[float, str] = None

    tight_layout: bool = False
    figsize: Optional[Tuple[int, int]] = (10, 6)

    x_ticks_rotation: Optional[int] = 45
    y_ticks_rotation: Optional[int] = None


@dataclasses.dataclass
class Hist2DPlotOptions(CommonPlotOptions):
    """
    Contains a set of options for displaying a bivariative histogram plot.

    :attr x_label_key: A key for x-axis values
    :attr x_label_name: A title for x-axis
    :attr y_label_key: An optional key for y-axis (If None, bar plot will use count of x-axis values)
    :attr y_label_name: A title for y-axis
    :attr x_lim: X-axis limits
    :attr y_lim: Y-axis limits
    :attr bins: Generic bin parameter that can be the name of a reference rule, the number of bins, or the breaks of the bins.
    :attr kde: If True, will display a kernel density estimate
    :attr stat: Aggregate statistic to compute in each bin. ("count", "frequency", "probability", "percent" or "density")
    :attr individual_plots_key: If None, the data will be displayed in a single plot.
                                If not None, will create a separate plot for each unique value of this column
    :attr individual_plots_max_cols: Sets the maximum number of columns to plot in the individual plots
    :attr labels_key: If you want to display multiple classes on same plot use this property to indicate column
    :attr labels_palette: Setting this allows you to control the colors of the bars of each label: { "train": "royalblue", "val": "red", "test": "limegreen" }
    :attr tight_layout: If True enables more compact layout of the plot
    :attr figsize: Size of the figure
    :attr sharey: Controls sharing of properties among y-axis (title, ticks, y_lim, ...). bool or {'none', 'all', 'row', 'col'}
    """

    x_label_key: str
    x_label_name: str

    y_label_key: Optional[str] = None
    y_label_name: Optional[str] = None

    weights: Optional[str] = None

    x_lim: Tuple[float, float] = None
    y_lim: Tuple[float, float] = None

    bins: Optional[int] = None
    kde: bool = False
    stat: str = "count"

    individual_plots_key: str = None
    individual_plots_max_cols: int = None

    labels_key: Optional[str] = None
    labels_name: Optional[str] = None
    labels_palette: Optional[Mapping] = None

    tight_layout: bool = False
    figsize: Optional[Tuple[int, int]] = (10, 6)

    x_ticks_rotation: Optional[int] = 45
    y_ticks_rotation: Optional[int] = None

    sharey: Union[bool, str] = False


@dataclasses.dataclass
class KDEPlotOptions(CommonPlotOptions):
    """
    Contains a set of options for displaying a kde histogram plot.

    :attr x_label_key: A key for x-axis values
    :attr x_label_name: A title for x-axis
    :attr y_label_key: An optional key for y-axis (If None, bar plot will use count of x-axis values)
    :attr y_label_name: A title for y-axis
    :attr x_lim: X-axis limits
    :attr y_lim: Y-axis limits
    :attr individual_plots_key: If None, the data will be displayed in a single plot.
                                If not None, will create a separate plot for each unique value of this column
    :attr individual_plots_max_cols: Sets the maximum number of columns to plot in the individual plots
    :attr labels_key: If you want to display multiple classes on same plot use this property to indicate column
    :attr labels_palette: Setting this allows you to control the colors of the bars of each label: { "train": "royalblue", "val": "red", "test": "limegreen" }
    :attr tight_layout: If True enables more compact layout of the plot
    :attr figsize: Size of the figure
    :attr common_norm:  If True, scale each conditional density by the number of observations such that the total area under all densities sums to 1.
                        Otherwise, normalize each density independently
    :attr bw_adjust:    Multiply the bandwidth by this value
    :attr fill:         If True, will fill the area under the curve
    :attr alpha:        Set the alpha value of the fill. Used only when fill==True
    :attr sharey: Controls sharing of properties among y-axis (title, ticks, y_lim, ...). bool or {'none', 'all', 'row', 'col'}
    """

    x_label_key: str
    x_label_name: str

    y_label_key: Optional[str] = None
    y_label_name: Optional[str] = None

    weights: Optional[str] = None

    x_lim: Tuple[float, float] = None
    y_lim: Tuple[float, float] = None

    individual_plots_key: str = None
    individual_plots_max_cols: int = None

    labels_key: Optional[str] = None
    labels_name: Optional[str] = None
    labels_palette: Optional[Mapping] = None

    tight_layout: bool = False
    figsize: Optional[Tuple[int, int]] = (10, 6)

    common_norm: bool = True
    bw_adjust: Optional[float] = None

    x_ticks_rotation: Optional[int] = 45
    y_ticks_rotation: Optional[int] = None

    fill: bool = False
    alpha: float = 0.1

    sharey: Union[bool, str] = False


@dataclasses.dataclass
class ScatterPlotOptions(CommonPlotOptions):
    """
    Contains a set of options for displaying a bivariative histogram plot.

    :attr x_label_key: A key for x-axis values
    :attr x_label_name: A title for x-axis
    :attr y_label_key: An optional key for y-axis (If None, bar plot will use count of x-axis values)
    :attr y_label_name: A title for y-axis
    :attr x_lim: X-axis limits
    :attr y_lim: Y-axis limits
    :attr bins: Generic bin parameter that can be the name of a reference rule, the number of bins, or the breaks of the bins.
    :attr kde: If True, will display a kernel density estimate
    :attr individual_plots_key: If None, the data will be displayed in a single plot.
                                If not None, will create a separate plot for each unique value of this column
    :attr individual_plots_max_cols: Sets the maximum number of columns to plot in the individual plots
    :attr labels_key: If you want to display multiple classes on same plot use this property to indicate column
    :attr labels_palette: Setting this allows you to control the colors of the bars of each label: { "train": "royalblue", "val": "red", "test": "limegreen" }
    :attr tight_layout: If True enables more compact layout of the plot
    :attr figsize: Size of the figure
    :attr sharey: Controls sharing of properties among y-axis (title, ticks, y_lim, ...). bool or {'none', 'all', 'row', 'col'}
    """

    x_label_key: str
    x_label_name: str

    y_label_key: str
    y_label_name: str

    x_lim: Tuple[float, float] = None
    y_lim: Tuple[float, float] = None

    individual_plots_key: str = None
    individual_plots_max_cols: int = None

    labels_key: Optional[str] = None
    labels_name: Optional[str] = None
    labels_palette: Optional[Mapping] = None

    tight_layout: bool = False
    figsize: Optional[Tuple[int, int]] = (10, 6)

    x_ticks_rotation: Optional[int] = 45
    y_ticks_rotation: Optional[int] = None

    sharey: Union[bool, str] = False


@dataclasses.dataclass
class HeatmapOptions(CommonPlotOptions):
    x_label_name: str
    y_label_name: str
    xticklabels: Union[bool, List[str]]
    yticklabels: Union[bool, List[str]]
    cbar: bool
    cmap: str
    annot: bool
    square: bool
    tight_layout: bool = False
    figsize: Optional[Tuple[int, int]] = (10, 6)
    fmt: str = None
    x_ticks_rotation: int = 0
    y_ticks_rotation: int = 0


@dataclasses.dataclass
class FigureRenderer(CommonPlotOptions):
    """Contains a set of options for displaying a pre-defined figure."""

    pass


class PlotRenderer(ABC):
    @abstractmethod
    def render(self, df: pd.DataFrame, options: CommonPlotOptions):
        ...
