import os
import logging
from datetime import datetime
from typing import Optional, List, Dict

import data_gradients
from data_gradients.assets import assets
from data_gradients.utils.pdf_writer import PDFWriter, ResultsContainer
from data_gradients.utils.utils import write_json, copy_files_by_list

logger = logging.getLogger(__name__)


class SummaryWriter:
    """Manager responsible for logging the Report (e.g. PDF), feature stats, errors and config cache."""

    def __init__(self, report_title: str, report_subtitle: Optional[str] = None, log_dir: Optional[str] = None):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = report_title.replace(" ", "_")

        # DIRECTORIES
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs", self.run_name)
            logger.info(f"`log_dir` was not set, so the logs will be saved in {log_dir}")
        self.log_dir = log_dir  # Main logging directory. Latest run results will be saved here.
        self.archive_dir = os.path.join(log_dir, "archive_" + timestamp)  # A duplicate of the results will be archived here as well
        os.makedirs(self.archive_dir, exist_ok=True)

        # OUTPUT PATH
        self.report_archive_path = os.path.join(self.archive_dir, "Report.pdf")
        self.summary_archive_path = os.path.join(self.archive_dir, "summary.json")
        self.errors_path = os.path.join(self.archive_dir, "errors.json")

        report_subtitle = report_subtitle or datetime.strftime(datetime.now(), "%m:%H %B %d, %Y")
        self._pdf_writer = PDFWriter(title=report_title, subtitle=report_subtitle, html_template=assets.html.doc_template)

        # DATA TO SAVE
        self._metadata = {"__version__": data_gradients.__version__, "report_title": report_title, "report_subtitle": report_subtitle, "timestamp": timestamp}
        self._data_config_dict = {}
        self._pdf_summary = ResultsContainer()
        self._features_stats: List[Dict[str, Dict]] = []
        self._errors: List[Dict[str, List[str]]] = []

    def set_pdf_summary(self, pdf_summary: ResultsContainer):
        self._pdf_summary = pdf_summary

    def set_data_config(self, data_config_dict: Dict):
        self._data_config_dict = data_config_dict

    def add_feature_stats(self, title: str, stats: Dict[str, Dict]):
        self._features_stats.append({"title": title, "stats": stats})

    def add_error(self, title: str, error: List[str]):
        self._errors.append({"title": title, "error": error})

    def write(self):
        """Write all the data accumulated until now."""

        # SUMMARY
        summary_json = {"metadata": self._metadata, "data_config": self._data_config_dict, "errors": self._errors, "features": self._features_stats}
        write_json(path=self.summary_archive_path, json_dict=summary_json)

        # ERRORS
        if self._errors:  # Log errors in a specific file, if any were found
            logger.warning(
                f"{len(self._errors)}/{len(self._features_stats)} features could not be processed.\n"
                f"You can find more information about what happened in {self.errors_path}"
            )
            error_json = {"metadata": self._metadata, "errors": self._errors}
            write_json(path=self.errors_path, json_dict=error_json)

        # PDF
        self._pdf_writer.write(results_container=self._pdf_summary, output_filename=self.report_archive_path)

        # COPY ARCHIVE_DIR -> LOG_DIR
        copy_files_by_list(
            source_dir=self.archive_dir,
            dest_dir=self.log_dir,
            file_list=[os.path.basename(self.summary_archive_path), os.path.basename(self.report_archive_path)],
        )
