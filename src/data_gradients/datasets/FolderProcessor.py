import os
import logging
from typing import List, Tuple, Sequence

logger = logging.getLogger(__name__)

DEFAULT_IMG_EXTENSIONS = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm")


class ImageLabelFolderIterator:
    """Iterate over all image and label files in the provided directories."""

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        label_extension: Sequence[str],
        image_extension: Sequence[str] = DEFAULT_IMG_EXTENSIONS,
        verbose: bool = True,
    ):
        """
        :param images_dir:      The directory containing the images.
        :param labels_dir:      The directory containing the labels.
        :param label_extension: The extensions of the labels. Only the labels with these extensions will be considered.
        :param image_extension: The extensions of the images. Only the images with these extensions will be considered.
        :param verbose:         Whether to print extra messages.
        """

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.verbose = verbose
        self.image_extension = self._normalize_extension(image_extension or DEFAULT_IMG_EXTENSIONS)
        self.label_extension = self._normalize_extension(label_extension)
        self.images_with_labels_files = self.get_image_and_label_file_names(images_dir=images_dir, labels_dir=labels_dir)

    def _normalize_extension(self, extensions: List[str]) -> List[str]:
        """Ensure that all extensions are lower case and don't include the '.'"""
        return [ext.replace(".", "").lower() for ext in extensions]

    def get_image_and_label_file_names(self, images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
        """Gather all image and label files from the provided sub_dirs."""
        images_with_labels_files = []

        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"The image directory `{images_dir}` does not exist.")
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"The label directory `{labels_dir}` does not exist.")

        images_files, labels_files = self._get_file_names_in_folder(images_dir, labels_dir)
        matched_images_with_labels_files = self._match_file_names(images_files, labels_files)

        images_with_labels_files.extend(matched_images_with_labels_files)

        return images_with_labels_files

    def _get_file_names_in_folder(self, images_dir: str, labels_dir: str) -> Tuple[List[str], List[str]]:
        """Extracts the names of all image and label files in the provided folders."""
        image_files = [os.path.abspath(os.path.join(images_dir, f)) for f in os.listdir(images_dir) if self.is_image(filename=f)]
        label_files = [os.path.abspath(os.path.join(labels_dir, f)) for f in os.listdir(labels_dir) if self.is_label(filename=f)]
        return image_files, label_files

    def _match_file_names(self, all_images_file_names: List[str], all_labels_file_names: List[str]) -> List[Tuple[str, str]]:
        """Matches the names of image and label files."""

        image_file_base_names = {get_filename(file_name): file_name for file_name in all_images_file_names}
        label_file_base_names = {get_filename(file_name): file_name for file_name in all_labels_file_names}

        common_base_names = set(image_file_base_names.keys()) & set(label_file_base_names.keys())
        unmatched_image_files = set(image_file_base_names.keys()) - set(label_file_base_names.keys())
        unmatched_label_files = set(label_file_base_names.keys()) - set(image_file_base_names.keys())

        if self.verbose:
            for imagefile in unmatched_image_files:
                logger.warning(f"Image file {imagefile} does not have a matching label file. Hide this message by setting verbose=False.")
            for label_file in unmatched_label_files:
                logger.warning(f"Label file {label_file} does not have a matching image file. Hide this message by setting verbose=False.")

        return [(image_file_base_names[name], label_file_base_names[name]) for name in common_base_names]

    def is_image(self, filename: str) -> bool:
        """Check if the given file name refers to image."""
        return filename.split(".")[-1].lower() in self.image_extension

    def is_label(self, filename: str) -> bool:
        """Check if the given file name refers to image."""
        return filename.split(".")[-1].lower() in self.label_extension

    def __len__(self):
        return len(self.images_with_labels_files)

    def __getitem__(self, index):
        return self.images_with_labels_files[index]

    def __iter__(self):
        for image_label_file in self.images_with_labels_files:
            yield image_label_file


class ImageLabelConfigIterator:
    """Iterate over all image and label files in the provided directories."""

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        config_path: str,
        label_extension: Sequence[str],
        image_extension: Sequence[str] = DEFAULT_IMG_EXTENSIONS,
        verbose: bool = True,
    ):
        """
        :param images_dir:      The directory containing the images.
        :param labels_dir:      The directory containing the labels.
        :param label_extension: The extensions of the labels. Only the labels with this extensions will be considered.
        :param image_extension: The extensions of the images. Only the images with this extensions will be considered.
        :param verbose:         Whether to print extra messages.
        """
        self.images_with_labels_files = self.get_image_and_label_file_names(
            images_dir=images_dir,
            labels_dir=labels_dir,
            config_path=config_path,
            label_extension=label_extension,
            image_extension=image_extension,
            verbose=verbose,
        )

    def get_image_and_label_file_names(
        self,
        images_dir: str,
        labels_dir: str,
        config_path: str,
        image_extension: List[str],
        label_extension: List[str],
        verbose: bool,
    ) -> List[Tuple[str, str]]:
        image_label_folder_iterator = ImageLabelFolderIterator(
            images_dir=images_dir,
            labels_dir=labels_dir,
            image_extension=image_extension,
            label_extension=label_extension,
            verbose=verbose,
        )
        file_ids = self._load_file_ids(config_path=config_path)
        filename_to_image_label_pair = {get_filename(image_path): (image_path, label_path) for (image_path, label_path) in image_label_folder_iterator}

        images_with_labels_files = []
        for file_id in file_ids:
            if file_id in filename_to_image_label_pair:
                images_with_labels_files.append(filename_to_image_label_pair[file_id])
            elif verbose:
                logger.warning(
                    f"No file with `file_id={file_id}` found in `images_dir={images_dir}` and/or `labels_dir={labels_dir}`. "
                    f"Hide this message by setting `verbose=False`."
                )

        if images_with_labels_files == []:
            error_msg = (
                f"Out of {len(file_ids)} file ids found in `config_path={config_path}`, "
                f"no matching file found in `images_dir={images_dir}` and/or `labels_dir={labels_dir}`."
            )
            if not verbose:
                error_msg += "\nSet `verbose=True` for more information."
            raise RuntimeError(error_msg)
        elif len(images_with_labels_files) != len(file_ids) and verbose:
            logger.warning(
                f"Out of {len(file_ids)} file ids found in `config_path={config_path}`, "
                f"{len(images_with_labels_files)} were found in both `images_dir={images_dir}` and `labels_dir={labels_dir}`. "
                f"Hide this message by setting `verbose=False`."
            )

        return images_with_labels_files

    def _load_file_ids(self, config_path: str) -> List[str]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"The config file `{config_path}` does not exist.")

        with open(config_path, "r") as f:
            try:
                file_ids = f.read().split()
            except Exception as e:
                raise e(f"Could not properly parse `config_path={config_path}`") from e

        if file_ids == []:
            raise RuntimeError(f"`config_path={config_path}` is empty and contains no file IDs.")

        return file_ids

    def __len__(self):
        return len(self.images_with_labels_files)

    def __getitem__(self, index):
        return self.images_with_labels_files[index]

    def __iter__(self):
        for image_label_file in self.images_with_labels_files:
            yield image_label_file


def get_filename(file_name: str) -> str:
    return os.path.splitext(os.path.basename(file_name))[0]
