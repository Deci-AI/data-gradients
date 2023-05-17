import os
from logging import getLogger

from xhtml2pdf import pisa
import tempfile

from data_gradients.feature_extractors.result import FeaturesCollection
from data_gradients.writers.abstract_writer import AbstractWriter
from data_gradients.writers.html_writer import HTMLWriter
from data_gradients.reports.report_template import ReportTemplate

logger = getLogger(__name__)


class PDFWriter(AbstractWriter):
    """
    A writer that writes a report to a PDF file.
    In uses HTMLWriter internally to generated temporary HTML file and then converts it to PDF.
    """

    def __init__(self, output_file: str):
        """
        :param output_file: The output file to which the report should be written.
        """
        self.output_file = os.path.abspath(output_file)

    def write_report(self, results: FeaturesCollection, template: ReportTemplate) -> None:

        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as td:
            output_html_file = os.path.join(td, "report.html")
            HTMLWriter(output_html_file).write_report(results, template)

            with open(output_html_file, "r") as html_source:
                with open(self.output_file, "w+b") as pdf_dst:
                    status = pisa.CreatePDF(src=html_source,
                                            dest=pdf_dst,
                                            path=os.path.dirname(output_html_file),
                                            options={"enable-local-file-access": None}, verbose=True)

            # pdfkit.from_file(output_html_file, self.output_file, options={"enable-local-file-access": None}, verbose=True)
            logger.debug(f"PDF Report written to {self.output_file}. Status {status}")
