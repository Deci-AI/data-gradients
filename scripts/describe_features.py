from data_gradients.feature_extractors import common, object_detection, segmentation
import inspect


def section_name_to_md_link(name: str) -> str:
    """Convert a section name to markdown link.
    :param name: Name of the section, e.g. "1. Image Features"
    :return: Markdown link, e.g. "[1. Image Features](#1-image-features)"
    """
    link = name.lower().replace(" ", "-").replace(".", "")
    return f"[{name}](#{link})"


def class_to_github_url(class_obj: type) -> str:
    github_base_url = "https://github.com/Deci-AI/data-gradients/blob/master/src/"

    class_path = inspect.getmodule(class_obj).__name__
    module_path = class_path.replace(".", "/") + ".py"
    return github_base_url + module_path


tasks = ["Image Features", "Object Detection Features", "Segmentation Features"]
modules = [common, object_detection, segmentation]

CLASSES_TO_IGNORE = ["SummaryStats"]

# Placeholders for the text
feature_descriptions = ""
table_of_contents = "### List of Features\n\n"

# Iterate over modules
for task, module in zip(tasks, modules):

    # Add module to table of contents
    table_of_contents += f"- {section_name_to_md_link(task)}\n"

    # Add module title
    feature_descriptions += f"### {task}\n\n<br/>\n\n"

    # Iterate over classes in module
    class_objects = inspect.getmembers(module, inspect.isclass)
    class_objects = [(class_name, class_obj) for class_name, class_obj in class_objects if class_name not in CLASSES_TO_IGNORE]
    for i, (class_name, class_obj) in enumerate(class_objects):
        feature = class_obj()

        feature_title = f"{i+1}. {feature.title}"
        table_of_contents += f"    - {section_name_to_md_link(feature_title)}\n"
        feature_descriptions += f"#### {feature_title}\n\n"
        feature_descriptions += f"{feature.description}\n"
        feature_descriptions += f"*[source code]({class_to_github_url(class_obj)})*\n\n<br/>\n\n"

    # Add empty line between modules
    feature_descriptions += "\n"

# Combine table of contents and summary
summary = f"""## Features Description

This page focuses on the description of features.

If you are interested in using these features, there is a tutorial specifically about [Features Configuration](feature_configuration.md).

{table_of_contents}

{feature_descriptions}"""

with open("../documentation/feature_description.md", "w") as f:
    f.write(summary)
