from typing import Mapping, Sequence, Callable, Optional, Any, List
import json

import torch
from numpy import ndarray
from pygments import lexers, formatters, highlight
from torch import Tensor


class ContainerMapper:
    def __init__(self):
        self._mapper: Optional[Callable] = None
        self._route: List[str] = []

    def analyze(self, objs):
        if isinstance(objs, dict) or self.isjson(objs):
            self._get_dict_mapping(objs)
        elif isinstance(objs, Mapping):
            raise NotImplementedError
        elif isinstance(objs, Sequence):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _get_dict_mapping(self, objs):
        self._route = self._get_users_string(objs)
        self._mapper = self._dict_mapping

    def _dict_mapping(self, objs):
        for r in self._route:
            objs = objs[r]
        return objs

    def container_to_tensor(self, objs) -> Tensor:
        return self._mapper(objs)

    @staticmethod
    def _get_users_string(objs):
        targets = []
        res = container_mapping(objs, path="", targets=targets)
        map_for_printing = json.dumps(res, indent=5, ensure_ascii=False)
        colorful_json = highlight(map_for_printing, lexers.JsonLexer(), formatters.TerminalFormatter())
        print(colorful_json.replace("\"", ""))
        value = int(input("which one of the yellow items is your required data?\n"))
        print(f'Path for getting objects out of container: {targets[value]}')
        keys = [r.replace("'", "").replace('[', '').replace(']', '') for r in targets[value].split(']')][:-1]
        return keys

    @staticmethod
    def isjson(myjson):
        try:
            json.loads(myjson)
        except ValueError as e:
            return False
        else:
            return True


def container_mapping(obj: Any, path: str, targets: list):
    if isinstance(obj, Mapping):
        printable_map = {}
        for k, v in obj.items():
            printable_map[k] = container_mapping(v, path + f"['{k}']", targets)
    elif isinstance(obj, Sequence):
        printable_map = []
        for i, o in enumerate(obj):
            printable_map.append(f'{i}: ' + container_mapping(o, path + f"[{i}]", targets))
    elif isinstance(obj, torch.Tensor):
        printable_map = f"Tensor {numbers[len(targets) % len(numbers)]}"
        targets.append(path)
    elif isinstance(obj, ndarray):
        printable_map = f"ndarray {numbers[len(targets) % len(numbers)]}"
        targets.append(path)
    elif isinstance(obj, ndarray):
        printable_map = f"PIL Image {numbers[len(targets) % len(numbers)]}"
        targets.append(path)
    else:
        raise RuntimeError("unsupported object")
    return printable_map


numbers = ["⓪", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
