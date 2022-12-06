import numpy as np
from matplotlib import pyplot as plt


def create_bar_plot(ax, data, labels, x_label: str = "", y_label: str = "", title: str = "",
                    train: bool = True, width: float = 0.4, ticks_rotation: int = 270,
                    train_color: str = 'green', val_color: str = 'red'):

    number_of_labels = len(labels)
    ax.bar(x=np.arange(number_of_labels) - (width / 2 if train else -width / 2),
           height=data,
           width=width,
           label='train' if train else 'val',
           color=train_color if train else val_color)

    plt.xticks(ticks=[label for label in range(number_of_labels)],
               labels=labels,
               rotation=ticks_rotation)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
