from typing import Union, Tuple, List, Mapping, Dict

SupportedData = Union[Tuple, List, Mapping, Tuple, List]
JSONValue = Union[str, int, float, bool, None, Dict[str, Union["JSONValue", List["JSONValue"]]]]
JSONDict = Dict[str, JSONValue]
