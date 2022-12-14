import hydra
from omegaconf import DictConfig

from src.manager import AnalysisManager
from data_loaders.get_torch_loaders import train_data_iterator, val_data_iterator


# TODO: Maybe get hydra implementation inside analysis manager as parser
@hydra.main(version_base=None, config_path="config/", config_name='feature_extractors')
def main(cfg: DictConfig) -> None:
    da = AnalysisManager(cfg, train_data_iterator, val_data_iterator)
    da.run()


if __name__ == '__main__':
    main()
