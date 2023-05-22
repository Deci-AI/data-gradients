import sys
import unittest

from tests.unit_tests.average_brightness_test import AverageBrightnessTest


class CoreUnitTestSuiteRunner:
    def __init__(self):
        self.test_loader = unittest.TestLoader()
        self.unit_tests_suite = unittest.TestSuite()
        self._add_modules_to_unit_tests_suite()
        self.end_to_end_tests_suite = unittest.TestSuite()
        self._add_modules_to_end_to_end_tests_suite()
        self.test_runner = unittest.TextTestRunner(verbosity=3, stream=sys.stdout)

    def _add_modules_to_unit_tests_suite(self):
        """
        _add_modules_to_unit_tests_suite - Adds unit tests to the Unit Tests Test Suite
            :return:
        """
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(AverageBrightnessTest))


if __name__ == "__main__":
    unittest.main()
