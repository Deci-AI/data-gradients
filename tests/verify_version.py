import sys

import data_gradients

if __name__ == "__main__":

    ci_version = sys.argv[1]
    if ci_version == data_gradients.__version__:
        sys.exit(0)
    else:
        print(f"wrong version definition:\nCI version: {ci_version}\ndata_gradients.__version__: {data_gradients.__version__}")
        sys.exit(1)
