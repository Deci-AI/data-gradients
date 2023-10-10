import sys
import unittest

from unit_tests.average_brightness_test import AverageBrightnessTest
from unit_tests.feature_extractors.detection.test_bounding_boxes_area import TestComputeHistogram
from unit_tests.dataset_output_mapper import TestImageConverter


class CoreUnitTestSuiteRunner:
    def __init__(self):
        self.test_loader = unittest.TestLoader()
        self.unit_tests_suite = unittest.TestSuite()
        self._add_modules_to_unit_tests_suite()
        self.test_runner = unittest.TextTestRunner(verbosity=3, stream=sys.stdout)

    def _add_modules_to_unit_tests_suite(self):
        """
        _add_modules_to_unit_tests_suite - Adds unit tests to the Unit Tests Test Suite
            :return:
        """
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(AverageBrightnessTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestComputeHistogram))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestImageConverter))


if __name__ == "__main__":
    unittest.main()
