from typing import Union, Tuple, List, Mapping, Dict, Type

from data_gradients.feature_extractors import AbstractFeatureExtractor

SupportedDataType = Union[Tuple, List, Mapping]
JSONValue = Union[str, int, float, bool, None, Dict[str, Union["JSONValue", List["JSONValue"]]]]
JSONDict = Dict[str, JSONValue]
FeatureExtractorsType = Union[
    List[Union[str, AbstractFeatureExtractor, Type[AbstractFeatureExtractor]]],
    str,
    AbstractFeatureExtractor,
    Type[AbstractFeatureExtractor],
]
