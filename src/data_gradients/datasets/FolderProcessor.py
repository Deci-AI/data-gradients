import os
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_IMG_EXTENSIONS = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm")


class ImageLabelFilesIterator:
    def __init__(
        self,
        *,
        root_dir: Optional[str] = None,
        images_subdir: str,
        labels_subdir: str,
        image_extension: Optional[List[str]] = None,
        label_extension: List[str],
        verbose: bool = True,
    ):
        self.images_subdir = images_subdir
        self.labels_subdir = labels_subdir
        self.root_dir = root_dir
        self.verbose = verbose
        self.image_extension = self._clean_extension(image_extension or DEFAULT_IMG_EXTENSIONS)
        self.label_extension = self._clean_extension(label_extension)
        self.images_with_labels_files = self.get_image_and_label_file_names(images_subdir=images_subdir, labels_subdir=labels_subdir)

    def _clean_extension(self, extensions: List[str]) -> List[str]:
        """Removes the "." from any extension from a list."""
        return [ext.replace(".", "") for ext in extensions]

    def _validate_user_specified_dirs(self, sub_dirs: List[str]):
        """Use user specified training and validation split_directories."""
        for dir_name in sub_dirs:
            full_dir_path = os.path.join(self.root_dir, dir_name)
            if not os.path.exists(full_dir_path):
                raise ValueError(f"The directory {full_dir_path} does not exist.")

    def get_image_and_label_file_names(self, images_subdir: str, labels_subdir: str) -> List[Tuple[str, str]]:
        """This function gathers all image and label files from the provided sub_dirs."""
        images_with_labels_files = []

        images_dir = images_subdir if self.root_dir is None else os.path.join(self.root_dir, images_subdir)
        labels_dir = labels_subdir if self.root_dir is None else os.path.join(self.root_dir, labels_subdir)

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
        base_name = lambda file_name: os.path.splitext(os.path.basename(file_name))[0]

        image_file_base_names = {base_name(file_name): file_name for file_name in all_images_file_names}
        label_file_base_names = {base_name(file_name): file_name for file_name in all_labels_file_names}

        common_base_names = set(image_file_base_names.keys()) & set(label_file_base_names.keys())
        unmatched_image_files = set(image_file_base_names.keys()) - set(label_file_base_names.keys())
        unmatched_label_files = set(label_file_base_names.keys()) - set(image_file_base_names.keys())

        if self.verbose:
            for imagefile in unmatched_image_files:
                logger.warning(f"Image file {imagefile} does not have a matching label file. Hide this message by setting verbose=False.")
            for label_file in unmatched_label_files:
                logger.warning(f"Warning: Label file {label_file} does not have a matching image file. Hide this message by setting verbose=False.")

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
