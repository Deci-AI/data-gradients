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
    """
    A container for the results of the analysis.
    dived to sections and features.
    """
    def __init__(self):
        self.sections = []

    def add_section(self, section: Section):
        self.sections.append(section)


class PDFWriter:
    """
    This class is responsible for generating the PDF file.
    It uses the pisa library to generate the PDF file.

    The PDF file is generated based on HTML templates (document, section and feature templates).
    """
    def __init__(self, title: str, subtitle: str, doc_template: str = assets.html.doc_template, section_template: str = assets.html.section_template,
                 feature_template: str = assets.html.feature_template, logo_path: str = assets.image.logo):
        """
        :param title: The title of the PDF document.
        :param subtitle: The subtitle of the PDF document.
        :param doc_template: The path to the document template.
        :param section_template: The path to the section template.
        :param feature_template: The path to the feature template.
        :param logo_path: The path to the logo image.
        """
        self.title = title
        self.subtitle = subtitle
        self.doc_template = doc_template
        self.section_template = section_template
        self.feature_template = feature_template
        self.logo_path = logo_path

    def _get_html(self, results_container: ResultsContainer):
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
        """
        :param results_container: The results container containing the sections and features.
        :param output_filename: The path to the output file.
        """
        if not output_filename.endswith("pdf"):
            raise RuntimeError("filename must end with .pdf")

        doc = self._get_html(results_container)
        with open(output_filename, "w+b") as result_file:
            pisa.CreatePDF(doc, dest=result_file)
