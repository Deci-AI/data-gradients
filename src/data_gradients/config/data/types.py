from typing import Union, Tuple, List, Mapping, Dict

SupportedDataType = Union[Tuple, List, Mapping]
JSONValue = Union[str, int, float, bool, None, Dict[str, Union["JSONValue", List["JSONValue"]]]]
JSONDict = Dict[str, JSONValue]
