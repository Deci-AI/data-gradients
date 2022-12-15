import concurrent
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Iterator, Iterable, Union, List

import hydra
import tqdm

from src.managers.abstract_manager import AnalysisManagerAbstract
from src.preprocess import SegmentationPreprocessor

debug_mode = False


class SegmentationAnalysisManager(AnalysisManagerAbstract):
    TASK = "semantic_segmentation"

    def __init__(self, *, num_classes: int,
                 train_data: Union[Iterable, Iterator],
                 ignore_labels: List = None,
                 val_data: Optional[Union[Iterable, Iterator]] = None,
                 ):
        super().__init__(train_data, val_data)
        self._task = self.TASK

        # Task Data Preprocessor
        self._preprocessor = SegmentationPreprocessor()
        self._preprocessor.number_of_classes = num_classes
        self._preprocessor.ignore_labels = ignore_labels

        self._parse_cfg()

    def _parse_cfg(self):
        hydra.initialize(config_path="../../config/", job_name="", version_base="1.1")
        self._cfg = hydra.compose(config_name=self._task)
        # TODO: Needs to disable strict mode
        self._cfg.number_of_classes = self._preprocessor.number_of_classes
        self._cfg.ignore_labels = self._preprocessor.ignore_labels

    def execute(self):
        pbar = tqdm.tqdm(desc='Working on batch #')
        train_batch = 0

        while True:
            if train_batch > 2:
                break
            try:
                batch_data = self._get_batch(self._train_iter)
            except StopIteration:
                break
            else:
                futures = [self._threads.submit(extractor.execute, batch_data) for extractor in
                           self._train_extractors]

            if not self._train_only:
                try:
                    batch_data = self._get_batch(self._val_iter)
                except StopIteration:
                    self._train_only = True
                else:
                    futures += [self._threads.submit(extractor.execute, batch_data) for extractor in
                                self._val_extractors]

            # Wait for all threads to finish
            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            pbar.update()
            train_batch += 1
