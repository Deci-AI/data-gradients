import inspect
from abc import ABC, ABCMeta


class InitAwareReprMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)

        # Extract the parameter names of the __init__ method in a clear manner
        param_names = list(inspect.signature(cls.__init__).parameters.keys())[1:]

        # Map these names to the provided arguments
        instance._init_params = {**dict(zip(param_names, args)), **kwargs}

        return instance


class InitAwareRepr(ABC, metaclass=InitAwareReprMeta):
    _init_params: dict  # Defined in the metaclass `InitAwareReprMeta`

    def __repr__(self):
        params = ", ".join([f"{k}={v!r}" for k, v in self._init_params.items()])
        return f"{self.__class__.__name__}({params})"
