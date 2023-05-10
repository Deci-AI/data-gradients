import os
import unittest

import pkg_resources

from data_gradients.utils.common.assets_container import assets, AssetNotFoundException


class TestAssets(unittest.TestCase):

    def setUp(self):
        self.asset_dir = pkg_resources.resource_filename("assets", "")

    def test_text_asset(self):
        self.assertEqual(assets.text.test, 'hello world!')

    def test_text_asset_not_found(self):
        with self.assertRaises(AssetNotFoundException):
            assets.text.nonexistent

    def test_html_asset(self):
        self.assertIn('<!DOCTYPE', assets.html.test)

    def test_html_asset_not_found(self):
        with self.assertRaises(AssetNotFoundException):
            assets.html.nonexistent

    def test_css_asset(self):
        self.assertEqual(assets.css.test, os.path.join(self.asset_dir, 'css', 'test.css'))

    def test_css_asset_not_found(self):
        with self.assertRaises(AssetNotFoundException):
            assets.css.nonexistent

    def test_image_asset(self):
        self.assertEqual(assets.image.logo, os.path.join(self.asset_dir, 'images', 'logo.png'))

    def test_image_asset_not_found(self):
        with self.assertRaises(AssetNotFoundException):
            assets.image.nonexistent


if __name__ == '__main__':
    unittest.main()
