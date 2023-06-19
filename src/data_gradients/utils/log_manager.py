import os
import logging
from datetime import datetime
from typing import Optional, List, Dict

from data_gradients.assets import assets
from data_gradients.utils.json_writer import MAIN_CACHE_DIR, log_errors, log_features
from data_gradients.utils.pdf_writer import PDFWriter, ResultsContainer
from data_gradients.utils.utils import copy_files_by_list

logger = logging.getLogger(__name__)


class LogManager:
    def __init__(self, report_title: str, report_subtitle: Optional[str] = None, log_dir: Optional[str] = None):
        # Static parameters
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs", report_title.replace(" ", "_"))
            logger.info(f"`log_dir` was not set, so the logs will be saved in {log_dir}")

        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = log_dir  # Main logging directory. Latest run results will be saved here.
        self.archive_dir = os.path.join(log_dir, "archive_" + session_id)  # A duplicate of the results will be archived here as well
        os.makedirs(self.archive_dir, exist_ok=True)

        self.report_archive_path = os.path.join(self.archive_dir, "Report.pdf")
        self.log_archive_path = os.path.join(self.archive_dir, "summary.json")
        self.log_errors_path = os.path.join(self.archive_dir, "errors.json")
        self.cache_path = os.path.join(MAIN_CACHE_DIR, report_title.replace(" ", "_") + ".json")

        self.pdf_writer = PDFWriter(title=report_title, subtitle=report_subtitle, html_template=assets.html.doc_template)

    def log(self, pdf_summary: ResultsContainer, features_stats: List[Dict[str, Dict]], errors: List[Dict[str, List[str]]]):
        if errors:  # Log errors in a specific file, if any were found
            logger.warning(
                f"{len(errors)}/{len(features_stats)} features could not be processed.\n"
                f"You can find more information about what happened in {self.log_errors_path}"
            )
            log_errors(errors_data=errors, path=self.log_errors_path)

        # Save to archive dir
        log_errors(errors_data=errors, path=self.log_archive_path)
        log_features(features_data=features_stats, path=self.log_archive_path)
        self.pdf_writer.write(results_container=pdf_summary, output_filename=self.report_archive_path)

        # Copy all from archive dir to log dir
        copy_files_by_list(
            source_dir=self.archive_dir,
            dest_dir=self.log_dir,
            file_list=[os.path.basename(self.log_archive_path), os.path.basename(self.report_archive_path)],
        )
