from data_gradients.logging.json_logger import JsonLogger
from data_gradients.logging.tensorboard_logger import TensorBoardLogger
from data_gradients.preprocess import PreprocessorAbstract
from data_gradients.utils import BatchData


class Logger:
    def __init__(self, tb_logger: TensorBoardLogger):
        self._tb_logger = tb_logger
        self._json_logger = JsonLogger(output_file_name="raw_data")

    def visualize(self, samples: BatchData):
        self._tb_logger.visualize(samples)

    def log(self, title_name: str, tb_data=None, json_data=None):
        if tb_data is not None:
            self._tb_logger.log(title_name, tb_data)
        if json_data is not None:
            self._json_logger.log(title_name, json_data)

    def log_meta_data(self, preprocessor: PreprocessorAbstract):
        if preprocessor.images_route is not None:
            self._json_logger.log("Get images out of dictionary", preprocessor.images_route)
        if preprocessor.labels_route is not None:
            self._json_logger.log("Get images out of dictionary", preprocessor.labels_route)

    def to_json(self):
        self._json_logger.write_to_json()

    def close(self):
        self._json_logger.close()
        self._tb_logger.close()

    def results_dir(self):
        return self._json_logger.logdir

    @staticmethod
    def get_title_name(class_name) -> str:
        title_name = class_name[0]
        for char in class_name[1:]:
            if char.isupper():
                title_name += " "
            title_name += char
        return title_name
