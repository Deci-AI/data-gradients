import os
import unittest

from data_gradients.assets import assets
from data_gradients.utils.pdf_writer import (
    ResultsContainer,
    Section,
    FeatureSummary,
    PDFWriter,
)


class PDFWriterTest(unittest.TestCase):
    def setUp(self):
        self.results_c = ResultsContainer()

        for section_index in range(4):
            section = Section(f"Section {section_index}")
            for feature_index in range(6):
                section.add_feature(
                    FeatureSummary(
                        f"Feature {feature_index}",
                        assets.text.lorem_ipsum,
                        assets.image.chart_demo,
                    )
                )
            self.results_c.add_section(section)

    def test_pdf_generation(self):

        html_writer = PDFWriter(
            title="Data Gradients",
            subtitle="test 2",
            html_template=assets.html.doc_template,
        )
        html_writer.write(self.results_c, "./out.pdf")
        self.assertTrue(os.path.exists("./out.pdf"))

    def tearDown(self):
        os.remove("./out.pdf")


if __name__ == "__main__":
    unittest.main()
