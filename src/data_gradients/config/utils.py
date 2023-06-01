import os.path
from typing import Optional, Dict, Any, List, Tuple

import hydra
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from data_gradients.feature_extractors import AbstractFeatureExtractor
from data_gradients.common.factories import FeatureExtractorsFactory, ListFactory

OmegaConf.register_new_resolver("merge", lambda x, y: x + y)


def load_report_feature_extractors(
    config_name: str, config_dir: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, List[AbstractFeatureExtractor]]:
    """Load and instantiate extractors from a Hydra configuration file.

    :param config_name: Name of the Hydra configuration file to load.
    :param config_dir:  Directory where the Hydra configuration file is located.
                        Defaults to `None`, which means that the default configuration directory of the package will be used.
    :param overrides:   Dictionary with overrides for the configuration file.
                        Defaults to `None`, which means no overrides will be applied.
    :return:            A list of instantiated feature extractors.
    """
    cfg = load_config(config_name=config_name, config_dir=config_dir, overrides=overrides)
    grouped_feature_extractors = {}
    for section in cfg["report_sections"]:
        section_name, feature_extractors = section["name"], section["features"]
        grouped_feature_extractors[section_name] = ListFactory(FeatureExtractorsFactory()).get(feature_extractors)
    return grouped_feature_extractors


def load_config(config_name: str, config_dir: str, overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
    """Load a Hydra configuration file and instantiate it.

    :param config_name: Name of the Hydra configuration file to load.
    :param config_dir:  Directory where the Hydra configuration file is located. By default, uses the package config directory.
    :param overrides:   Dictionary with overrides for the configuration file. By default, no overrides will be applied.
    :return:            An instantiated configuration object.
    """

    config_dir = config_dir or os.path.dirname(__file__)
    overrides = overrides or {}

    dotlist_overrides = dict_to_dotlist_overrides(overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name=config_name, overrides=dotlist_overrides)
    return hydra.utils.instantiate(cfg)


def dict_to_dotlist_overrides(dict_params: Dict[str, Any]) -> List[str]:
    """Convert a dictionary to a list of strings in format 'path.to.key=value', similar the hydra command line overrides.

    >>> dict_to_dotlist({'experiment_name': 'adam', 'model': {'type': 'resnet', 'depth': 18, 'num_classes': 10}})
    ['experiment_name=adam','model.type=resnet','model.depth=18','model.num_classes=10']

    :param dict_params: The dictionary to convert.
    :return:            List of strings in format 'path.to.key=value', similar the hydra command line overrides.
    """
    out = []
    for key, value in dict_to_dotlist(dict_params):
        if value is None:
            out.append(f"{key}=null")
        else:
            out.append(f"{key}={value}")
    return out


def dict_to_dotlist(dict_params: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Convert a dictionary to a list of dot-separated key-value pairs.

    This function takes a dictionary as input and converts it to a list of key-value pairs, where each key is a
    dot-separated string representing a nested dictionary key, and the value is the corresponding value in the
    dictionary.

    >>> dict_to_dotlist({'experiment_name': 'adam', 'model': {'type': 'resnet', 'depth': 18, 'num_classes': 10}})
    [('experiment_name', 'adam'), ('model.type','resnet'), ('model.depth', 18), ('model.num_classes', 10)]

    :param dict_params: The dictionary to convert.
    :return: A list of key-value pairs, where each key is a dot-separated string and each value is a value from the input dictionary.
    """

    dotlist_params = []
    for key, value in dict_params.items():
        if isinstance(value, dict):
            nested_params = dict_to_dotlist(value)
            for nested_key, nested_value in nested_params:
                dotlist_params.append((f"{key}.{nested_key}", nested_value))
        else:
            dotlist_params.append((key, value))
    return dotlist_params
