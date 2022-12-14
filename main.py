import hydra
from omegaconf import DictConfig

from manager import AnalysisManager
from data_loaders.get_torch_loaders import train_data_iterator, val_data_iterator


@hydra.main(version_base=None, config_path=".", config_name='feature_extractors')
def main(cfg: DictConfig) -> None:
    da = AnalysisManager(cfg, train_data_iterator, val_data_iterator)
    da.run()


if __name__ == '__main__':
    main()
