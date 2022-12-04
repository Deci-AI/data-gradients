import argparse
from manager import AnalysisManager
from create_torch_loaders import train_dataloader, val_dataloader


def main(args):
    da = AnalysisManager(args)
    da.build()
    da.execute(train_dataloader, val_dataloader)
    da.post_process()
    da.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Analysis tool by Deci.ai")
    parser.add_argument('--yaml-path', default='feature_extractors.yaml', help='path to yaml file')
    parser.add_argument('--task', choices=['object-detection', 'semantic-segmentation', 'classification',
                                           'instance-segmentation'],
                        default='semantic-segmentation', required=False, help='Choose your dataset type')
    main(parser.parse_args())
