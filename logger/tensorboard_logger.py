import datetime as dt
from collections import OrderedDict

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
