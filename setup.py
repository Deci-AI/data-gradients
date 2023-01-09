# coding: utf-8

"""
    Deci Dataset Analyzer
"""

from setuptools import setup
from setuptools import find_packages

README_LOCATION = "README.md"
REQ_LOCATION = "requirements.txt"
INIT_FILE = "src/__init__.py"


def readme():
    """print long description"""
    with open(README_LOCATION, encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open(REQ_LOCATION, encoding="utf-8") as f:
        requirements = f.read().splitlines()
        return [r for r in requirements if not r.startswith("--") and not r.startswith("#")]


setup(name="super-gradients",
      description="SuperGradients",
      author="Deci AI",
      author_email="rnd@deci.ai",
      url="https://deci-ai.github.io/super-gradients/welcome.html",
      keywords=["Deci", "AI", "Training", "Deep Learning", "Computer Vision", "PyTorch", "SOTA", "Recipes", "Pre Trained", "Models"],
      install_requires=get_requirements(),
      packages=find_packages(where="./src"),
      package_dir={"": "src"},
      package_data={"super_gradients.recipes": ["*.yaml", "**/*.yaml"],
                    "super_gradients.common": ["auto_logging/auto_logging_conf.json"],
                    "super_gradients.examples": ["*.ipynb", "**/*.ipynb"],
                    "super_gradients": ["requirements.txt", "requirements.pro.txt"],
                    },
      long_description=readme(),
      long_description_content_type="text/markdown",
      )