import pkg_resources

from data_gradients.assets.assets_container import Assets, AssetNotFoundException

assets = Assets(pkg_resources.resource_filename("data_gradients.assets", ""))

__all__ = ["assets", "AssetNotFoundException"]
