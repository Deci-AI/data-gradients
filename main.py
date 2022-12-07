import hydra
from omegaconf import DictConfig

from manager import AnalysisManager
from data_loaders.get_torch_loaders import DataLoaders


@hydra.main(version_base=None, config_path=".", config_name='feature_extractors')
def main(cfg: DictConfig) -> None:
    train_dataloader, val_dataloader = DataLoaders().get_dataloader(dataset='bdd')
    train_data_iterator, val_data_iterator = iter(train_dataloader), iter(val_dataloader)

    da = AnalysisManager(cfg, train_data_iterator, val_data_iterator)
    da.run()

if __name__ == '__main__':
    main()
