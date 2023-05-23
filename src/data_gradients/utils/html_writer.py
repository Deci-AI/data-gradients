from dataclasses import dataclass

from xhtml2pdf import pisa

from data_gradients.utils.common.assets_container import assets

FEATURE_NAME_KEY = "{{feature_name}}"
FEATURE_DESCRIPTION_KEY = "{{feature_description}}"
FEATURE_IMAGE_PATH_KEY = "{{feature_image_path}}"
SECTION_NAME_KEY = "{{section_name}}"
FEATURES_KEY = "{{features}}"
SECTIONS_KEY = "{{sections}}"
TITLE_KEY = "{{title}}"
SUBTITLE_KEY = "{{subtitle}}"
LOGO_PATH_KEY = "{{logo_path}}"


@dataclass
class Feature:
    name: str
    description: str
    image_path: str


class Section:
    def __init__(self, section_name):
        self.section_name = section_name
        self.features = []

    def add_feature(self, feature: Feature):
        self.features.append(feature)


class ResultsContainer:

    def __init__(self):
        self.sections = []

    def add_section(self, section: Section):
        self.sections.append(section)


class HtmlWriter:

    def __init__(self, title: str, subtitle: str, doc_template: str = assets.html.doc_template, section_template: str = assets.html.section_template,
                 feature_template: str = assets.html.feature_template, logo_path: str = assets.image.logo):
        self.title = title
        self.subtitle = subtitle
        self.doc_template = doc_template
        self.section_template = section_template
        self.feature_template = feature_template
        self.logo_path = logo_path

    def get_html(self, results_container: ResultsContainer):
        sections_html = ""
        for section_index, section in enumerate(results_container.sections):

            section_name = f"{section_index + 1}.\t{section.section_name}"
            features = section.features
            features_html = ""
            for feature_index, feature in enumerate(features):
                feature_name = f"{section_index + 1}.{feature_index + 1}.\t{feature.name}"
                feature_description = feature.description
                feature_image_path = feature.image_path

                features_html += self.feature_template.replace(FEATURE_NAME_KEY, feature_name) \
                    .replace(FEATURE_DESCRIPTION_KEY, feature_description) \
                    .replace(FEATURE_IMAGE_PATH_KEY, feature_image_path)

            sections_html += self.section_template.replace(SECTION_NAME_KEY, section_name) \
                .replace(FEATURES_KEY, features_html)

        doc = self.doc_template.replace(SECTIONS_KEY, sections_html).replace(TITLE_KEY, self.title) \
            .replace(SUBTITLE_KEY, self.subtitle).replace(LOGO_PATH_KEY, self.logo_path)
        return doc

    def write(self, results_container: ResultsContainer, output_filename: str):
        assert output_filename.endswith("html")
        doc = self.get_html(results_container)
        with open(output_filename, "w") as result_file:
            result_file.write(doc)

    def write_PDF(self, results_container: ResultsContainer, output_filename: str):
        assert output_filename.endswith("pdf")
        doc = self.get_html(results_container)
        with open(output_filename, "w+b") as result_file:
            pisa.CreatePDF(doc, dest=result_file)
