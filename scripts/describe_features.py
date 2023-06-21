from data_gradients.feature_extractors import common, object_detection, segmentation
import inspect


def section_name_to_md_link(name):
    return name.lower().replace(" ", "-").replace(".", "")


tasks = ["Image Features", "Object Detection Features", "Segmentation Features"]
modules = [common, object_detection, segmentation]

feature_descriptions = ""
table_of_contents = "## List of Features\n\n"


# Iterate over modules
for task, module in zip(tasks, modules):

    # Add module to table of contents
    table_of_contents += f"- [{task}](#{section_name_to_md_link(task)})\n"

    # Add module title
    feature_descriptions += f"## {task}\n\n"

    # Iterate over classes in module
    class_objects = inspect.getmembers(module, inspect.isclass)
    for i, (_, class_obj) in enumerate(class_objects):
        feature = class_obj()

        feature_title = f"{i}. {feature.title}"
        table_of_contents += f"    - [{feature_title}](#{section_name_to_md_link(f'{feature_title}')})\n"
        feature_descriptions += f"### {feature_title}\n\n"
        feature_descriptions += f"{feature.description}\n\n"

    # Add empty line between modules
    feature_descriptions += "\n"

# Combine table of contents and summary
summary = f"# Features\n\n{table_of_contents}\n\n{feature_descriptions}"

with open("features.md", "w") as f:
    f.write(summary)
