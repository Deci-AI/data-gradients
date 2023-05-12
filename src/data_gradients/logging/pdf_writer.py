import os
from logging import getLogger

import pdfkit
import tempfile

from data_gradients.feature_extractors.result import FeaturesResult
from data_gradients.logging.html_writer import HTMLWriter
from data_gradients.reports.report_template import ReportTemplate

logger = getLogger(__name__)


class PDFWriter:
    """
    A writer that writes a report to a PDF file.

    In uses HTMLWriter internally to generated temporary HTML file and then converts it to PDF using pdfkit.
    """

    def __init__(self, output_file: str):
        """
        :param output_file: The output file to which the report should be written.
        """
        self.output_file = os.path.abspath(output_file)

    def write_report(self, results: FeaturesResult, template: ReportTemplate) -> None:

        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as td:
            output_html_file = os.path.join(td, "report.html")
            HTMLWriter(output_html_file).write_report(results, template)

            pdfkit.from_file(output_html_file, self.output_file, options={"enable-local-file-access": None}, verbose=True)
            logger.debug(f"PDF Report written to {self.output_file}")
