import inspect
from abc import ABC, ABCMeta


class InitAwareReprMeta(ABCMeta):
    """
    Metaclass that intercepts the initialization of instances to store the provided
    arguments for initialization. This aids in producing clearer and more concise
    representations for instances using their initialization arguments.
    """

    def __call__(cls, *args, **kwargs):
        """
        Override the default call to the class, capturing the arguments provided
        during initialization and storing them in the _init_params attribute.

        :param args: Positional arguments passed during instance initialization.
        :param kwargs: Keyword arguments passed during instance initialization.
        :return: An instance of the class being initialized.
        """
        instance = super().__call__(*args, **kwargs)

        # Extract the parameter names of the __init__ method
        param_names = list(inspect.signature(cls.__init__).parameters.keys())[1:]

        # Map these names to the provided arguments
        instance._init_params = {**dict(zip(param_names, args)), **kwargs}

        return instance


class InitAwareRepr(ABC, metaclass=InitAwareReprMeta):
    """Abstract base class that provides a custom representation (__repr__) for instances
    using the arguments provided during initialization. This aids in clearer debugging and
    readability when printing or logging instances.

    :attr _init_params: Dictionary mapping the parameter names used during initialization to their provided values.

    Example:
        class ExampleClass(InitAwareRepr):
            def __init__(self, x, y):
                self.x = x
                self.y = y

        e = ExampleClass(1, 2)
        print(e)  # Outputs: ExampleClass(x=1, y=2)
    """

    _init_params: dict  # Defined in the metaclass `InitAwareReprMeta`

    def __repr__(self):
        """
        Override the default representation of the instance to use the parameters
        and values provided during initialization.

        :return: A string representation of the instance using initialization arguments.
        """
        params = ", ".join([f"{k}={v!r}" for k, v in self._init_params.items()])
        return f"{self.__class__.__name__}({params})"
