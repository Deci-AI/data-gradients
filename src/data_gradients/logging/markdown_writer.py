import os
import tempfile
from typing import Optional

from data_gradients.feature_extractors.result import FeaturesResult
from data_gradients.reports.report_template import ReportTemplate
from data_gradients.visualize.seaborn_renderer import SeabornRenderer


class MarkdownWriter:
    def __init__(self, output_file: str, images_subfolder: Optional[str] = None):
        """
        :param output_file: The output file to which the report should be written.
        :param images_subfolder: The subfolder in which the images should be stored. If None - images stored is the same folder as the report.
                                 If "img" - images stored in the "img" subfolder.
        """
        self.output_file = os.path.abspath(output_file)
        self.images_subfolder = images_subfolder

    def write_report(self, results: FeaturesResult, template: ReportTemplate) -> None:
        sns = SeabornRenderer()

        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)
        if self.images_subfolder is not None:
            os.makedirs(os.path.join(output_dir, self.images_subfolder), exist_ok=True)

        # Report generation
        with open(self.output_file, "w") as f:
            f.write("# Dataset result\n")

            for widget in template.widgets:
                plot = widget.to_figure(results, sns)


                widget_image_name = f"{widget.__class__.__name__}.png"
                relative_image_output_path = (
                    os.path.join(self.images_subfolder, widget_image_name).replace("\\", "/") if self.images_subfolder is not None else widget_image_name
                )

                plot.savefig(os.path.join(output_dir, relative_image_output_path))
                #plot.show()

                f.write("\n\n")
                f.write(f"![]({relative_image_output_path})")
                f.write("\n\n")


class HTMLWriter:
    def __init__(self, output_file: str, images_subfolder: Optional[str] = None):
        """
        :param output_file: The output file to which the report should be written.
        :param images_subfolder: The subfolder in which the images should be stored. If None - images stored is the same folder as the report.
                                 If "img" - images stored in the "img" subfolder.
        """
        self.output_file = os.path.abspath(output_file)
        self.images_subfolder = images_subfolder

    def write_report(self, results: FeaturesResult, template: ReportTemplate) -> None:
        sns = SeabornRenderer()

        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)
        if self.images_subfolder is not None:
            os.makedirs(os.path.join(output_dir, self.images_subfolder), exist_ok=True)

        # Report generation
        with open(self.output_file, "w") as f:
            f.write("<html>\n")
            f.write("<head>\n")
            f.write("<title>Dataset result</title>\n")
            f.write("</head>\n")
            f.write("<body>\n")

            for widget in template.widgets:
                plot = widget.to_figure(results, sns)

                widget_image_name = f"{widget.__class__.__name__}.png"
                relative_image_output_path = (
                    os.path.join(self.images_subfolder, widget_image_name).replace("\\", "/") if self.images_subfolder is not None else widget_image_name
                )

                plot.savefig(os.path.join(output_dir, relative_image_output_path))
                # plot.show()

                f.write(f"<h1>{widget.__class__.__name__}</h1>\n")
                f.write("<br>\n")
                f.write(f'<img src="{relative_image_output_path}" >\n')
                f.write("<br>\n")

            f.write("</body>\n")
            f.write("</html>\n")


class PDFWriter:
    def __init__(self, output_file: str):
        """
        :param output_file: The output file to which the report should be written.
        """
        self.output_file = os.path.abspath(output_file)

    def write_report(self, results: FeaturesResult, template: ReportTemplate) -> None:
        import pdfkit

        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as td:
            output_html_file = os.path.join(td, "report.html")
            HTMLWriter(output_html_file).write_report(results, template)

            pdfkit.from_file(output_html_file, self.output_file, options={'enable-local-file-access': None}, verbose=True)
