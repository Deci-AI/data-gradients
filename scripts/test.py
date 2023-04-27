def dict_to_dotlist(dict_params):
    dotlist_params = []
    for key, value in dict_params.items():
        if isinstance(value, dict):
            # Recursively convert nested dictionaries to dotlist format
            nested_params = dict_to_dotlist(value)
            for nested_key, nested_value in nested_params:
                dotlist_params.append((f"{key}.{nested_key}", nested_value))
        else:
            # Convert key-value pairs to dotlist format
            dotlist_params.append((key, value))
    return dotlist_params


params = {"experiment_name": "adam", "model": {"type": "resnet", "depth": 18, "num_classes": 10}}

print(params)
dotlist_params = dict_to_dotlist(params)
print(dotlist_params)
overrides = [f"{key}={value}" for key, value in dotlist_params]
print(overrides)
