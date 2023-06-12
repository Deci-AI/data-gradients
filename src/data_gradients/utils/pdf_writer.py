from dataclasses import dataclass

import seaborn
from jinja2 import Template
from xhtml2pdf import pisa

import data_gradients
from data_gradients.assets import assets


@dataclass
class FeatureSummary:
    name: str
    description: str
    image_path: str
    notice: str = None
    warning: str = None


class Section:
    def __init__(self, section_name):
        self.section_name = section_name
        self.features = []

    def add_feature(self, feature: FeatureSummary):
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

    def __init__(self, title: str, subtitle: str, html_template: str = assets.html.doc_template, logo_path: str = assets.image.logo, palette="pastel"):
        """
        :param title: The title of the PDF document.
        :param subtitle: The subtitle of the PDF document.
        :param html_template: The path to the document template.
        :param logo_path: The path to the logo image.
        """
        self.title = title
        self.subtitle = subtitle
        self.template = Template(source=html_template)
        self.logo_path = logo_path
        palette = seaborn.color_palette(palette=palette).as_hex()
        self.train_color = palette[0]
        self.val_color = palette[1]

    def write(self, results_container: ResultsContainer, output_filename: str):
        """
        :param results_container: The results container containing the sections and features.
        :param output_filename: The path to the output file.
        """
        if not output_filename.endswith("pdf"):
            raise RuntimeError("filename must end with .pdf")

        doc = self.template.render(
            title=self.title,
            subtitle=self.subtitle,
            results=results_container,
            version=data_gradients.__version__,
            train_color=self.train_color,
            val_color=self.val_color,
            logo=assets.image.logo,
            assets=assets,
        )

        with open(output_filename, "w+b") as result_file:
            pisa.CreatePDF(doc, dest=result_file)
