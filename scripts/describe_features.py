from data_gradients.feature_extractors import common, object_detection, segmentation
import inspect


def section_name_to_md_link(name: str):
    """Convert a section name to markdown link.
    :param name: Name of the section, e.g. "1. Image Features"
    :return: Markdown link, e.g. "[1. Image Features](#1-image-features)"
    """
    link = name.lower().replace(" ", "-").replace(".", "")
    return f"[{name}](#{link})"


github_base_url = "https://github.com/Deci-AI/data-gradients/blob/master/src/"


tasks = ["Image Features", "Object Detection Features", "Segmentation Features"]
modules = [common, object_detection, segmentation]

# Placeholders for the text
feature_descriptions = ""
table_of_contents = "## List of Features\n\n"

# Iterate over modules
for task, module in zip(tasks, modules):

    # Add module to table of contents
    table_of_contents += f"- {section_name_to_md_link(task)}\n"

    # Add module title
    feature_descriptions += f"## {task}\n\n"

    # Iterate over classes in module
    class_objects = inspect.getmembers(module, inspect.isclass)
    for i, (class_name, class_obj) in enumerate(class_objects):
        feature = class_obj()
        class_path = inspect.getmodule(class_obj).__name__
        class_github_url = github_base_url + class_path.replace(".", "/") + ".py"

        feature_title = f"{i+1}. {feature.title}"
        table_of_contents += f"    - {section_name_to_md_link(feature_title)}\n"
        feature_descriptions += f"### {feature_title}\n\n"
        feature_descriptions += f"{feature.description}\n"
        feature_descriptions += f"*[source code]({class_github_url})*\n\n"

    # Add empty line between modules
    feature_descriptions += "\n"

# Combine table of contents and summary
summary = f"# Features\n\n{table_of_contents}\n\n{feature_descriptions}"

with open("features.md", "w") as f:
    f.write(summary)