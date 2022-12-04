import datetime as dt
import os
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def get_dict_ordered_by_values(dictionary):
    dictionary = OrderedDict(sorted(dictionary.items()))


class TensorBoardLogger:

    def __init__(self):
        logdir = "logs/train_data/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.writer = SummaryWriter(log_dir=logdir)

    def graph_to_tensorboard(self, title, plot):
        self.writer.add_figure(title, plot)

    def text_to_tensorboard(self, title, text):
        self.writer.add_image(title, text)

    def close(self):
        self.writer.close()


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
