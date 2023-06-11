import os

import pkg_resources


class AssetNotFoundException(Exception):
    pass


class Asset:
    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path, "r") as f:
            return f.read()


class TextAssets:
    def __init__(self, asset_dir):
        self.asset_dir = os.path.join(asset_dir, "text")

    def __getattr__(self, name):
        asset_path = os.path.join(self.asset_dir, name + ".txt")

        if not os.path.exists(asset_path):
            raise AssetNotFoundException("Asset not found: {}".format(name))

        return Asset(asset_path).read()


class HTMLAssets:
    def __init__(self, asset_dir):
        self.asset_dir = os.path.join(asset_dir, "html")

    def __getattr__(self, name):
        asset_path = os.path.join(self.asset_dir, name + ".html")

        if not os.path.exists(asset_path):
            raise AssetNotFoundException("Asset not found: {}".format(name))

        return Asset(asset_path).read()


class CSSAssets:
    def __init__(self, asset_dir):
        self.asset_dir = os.path.join(asset_dir, "css")

    def __getattr__(self, name):
        asset_path = os.path.join(self.asset_dir, name + ".css")

        if not os.path.exists(asset_path):
            raise AssetNotFoundException("Asset not found: {}".format(name))

        return asset_path


class ImageAssets:
    def __init__(self, asset_dir):
        self.asset_dir = os.path.join(asset_dir, "images")

    def __getattr__(self, name):

        for ext in ["jpg", "jpeg", "png", "gif"]:
            asset_path = os.path.join(self.asset_dir, f"{name}.{ext}")

            if os.path.exists(asset_path):
                return asset_path
        raise AssetNotFoundException("Asset not found: {}".format(name))


class Assets:

    """
    Assets class to allow quick access to assets.
    usage:
       call assets.text.test to get content of assets/text/test.txt asset
       call assets.htm.test to get content of assets/html/test.html asset
       call assets.css.test to get full path of assets/text/test.css asset
       call assets.image.logo to get full path of assets/images/logo.png asset (supported image formats: jpg, jpeg, png, gif)
    """
    def __init__(self, asset_dir):
        self.asset_dir = asset_dir
        self._text_assets = TextAssets(asset_dir)
        self._image_assets = ImageAssets(asset_dir)
        self._css_assets = CSSAssets(asset_dir)
        self._html_assets = HTMLAssets(asset_dir)

    @property
    def text(self):
        return self._text_assets

    @property
    def image(self):
        return self._image_assets

    @property
    def css(self):
        return self._css_assets

    @property
    def html(self):
        return self._html_assets

