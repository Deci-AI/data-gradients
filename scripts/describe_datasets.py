import inspect
from data_gradients.datasets import detection as detection_datasets
from data_gradients.datasets import segmentation as segmentation_datasets
import re


def remove_first_indentation(s: str) -> str:
    # This regular expression matches lines that start with 4 spaces
    # and replaces them with nothing, thereby removing the first indentation.
    return re.sub(r"^    ", "", s, flags=re.MULTILINE)


def section_name_to_md_link(name: str) -> str:
    """Convert a section name to markdown link."""
    link = name.lower().replace(" ", "-").replace(".", "")
    return f"[{name}](#{link})"


def class_to_github_url(class_obj: type) -> str:
    github_base_url = "https://github.com/Deci-AI/data-gradients/blob/master/src/"
    class_path = inspect.getmodule(class_obj).__name__
    module_path = class_path.replace(".", "/") + ".py"
    return github_base_url + module_path


# Define the categories of datasets
categories = ["Detection Datasets", "Segmentation Datasets"]
modules = [detection_datasets, segmentation_datasets]

# Placeholder for the markdown content
dataset_descriptions = ""
table_of_contents = "## List of Datasets\n\n"

# Iterate over categories and corresponding modules
for category, module in zip(categories, modules):
    # Add category to table of contents
    table_of_contents += f"- {section_name_to_md_link(category)}\n"

    # Add category title
    dataset_descriptions += f"## {category}\n\n<br/>\n\n"

    # Get classes from module
    dataset_classes = inspect.getmembers(module, inspect.isclass)
    for i, (class_name, class_obj) in enumerate(dataset_classes):
        dataset_doc = class_obj.__doc__ if class_obj.__doc__ else "No description provided."
        # dataset_doc = '\n'.join([m.lstrip() for m in dataset_doc.split('\n')])
        dataset_doc = remove_first_indentation(dataset_doc)

        # Create dataset title and add to table of contents
        dataset_title = f"{i+1}. {class_name}"
        table_of_contents += f"    - {section_name_to_md_link(dataset_title)}\n"

        # Append dataset details to the markdown content
        dataset_descriptions += f"### {dataset_title}\n\n"
        dataset_descriptions += f"{dataset_doc}\n\n"
        dataset_descriptions += f"*[source code]({class_to_github_url(class_obj)})*\n\n<br/>\n\n"

    # Add empty line between categories
    dataset_descriptions += "\n"

# Combine table of contents and dataset descriptions
summary = f"""# Built-in Datasets

DataGradients offer a few basic datasets which can help you load your data without needing to provide any additional code.

These datasets contain only the basic functionalities.
They are meant to be used within SuperGradients and are not recommended to be used for training (No `transform` parameter available).

{table_of_contents}

{dataset_descriptions}"""

with open("../documentation/datasets.md", "w") as f:
    f.write(summary)
