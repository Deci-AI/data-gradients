import argparse
from manager import AnalysisManager
from data_loaders.get_torch_loaders import DataLoaders


def main(args):
    train_dataloader, val_dataloader = DataLoaders().get_dataloader(dataset='pp_human')
    da = AnalysisManager(args, train_dataloader, val_dataloader)
    da.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Analysis tool by Deci.ai")
    parser.add_argument('--yaml-path', default='feature_extractors.yaml', help='path to yaml file')
    parser.add_argument('--task', choices=['object-detection', 'semantic-segmentation', 'classification',
                                           'instance-segmentation'],
                        default='semantic-segmentation', required=False, help='Choose your dataset type')
    main(parser.parse_args())
