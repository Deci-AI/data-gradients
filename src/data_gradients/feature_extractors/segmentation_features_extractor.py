import typing
from typing import Mapping, Any, Optional, Union, List

import numpy as np
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, binary_erosion

from data_gradients.feature_extractors.features import SegmentationMaskFeatures


class SemanticSegmentationFeaturesExtractor:
    """
    Extracts features from a semantic segmentation mask.
    Important notice - this features extractor intended for use on semantic segmentation masks and is not
    suitable for instance segmentation masks.

    For extracting features from instance segmentation masks, use InstanceSegmentationFeaturesExtractor.
    """

    def __init__(self, ignore_labels: Optional[Union[int, List[int]]]):
        """

        :param ignore_labels: A label or list of labels that represents ignored class.
                              If None - all classes (including 0) in the mask will be analyzed.
                              If you want to exclude background label that has label 0 from the analysis,
                              set ignore_labels=0.
                              If you want to exclude unannotated region that has label 255 from the analysis,
                              set ignore_labels=255.
                              If you want to exclude unannotated region that has label 255 and the background label 0
                              from the analysis, set ignore_labels=[0,255].

        """
        self.ignore_labels = None
        if ignore_labels is not None:
            self.ignore_labels = tuple(ignore_labels) if isinstance(ignore_labels, typing.Iterable) else (ignore_labels,)

    def __call__(self, segmentation_mask: np.ndarray, shared_keys: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        """
        Extracts features from a single segmentation mask.

        :param segmentation_mask: An input segmentation mask of [H,W] shape.
        :param shared_keys: A dictionary of shared keys that will be added to the each row of the output.
                            For instance this may include image id or dataset split property that is shared
                            for every instance.

        :return: A dictionary of features
        """
        if not isinstance(segmentation_mask, np.ndarray):
            raise ValueError("segmentation_mask must be a numpy array. Got: {}".format(type(segmentation_mask)))


        segmentation_mask = segmentation_mask.copy()

        # label_dict holds the mapping between the new labels and the original labels of the mask
        # this is necessary because we can change the labels of the mask in order to remove background and ignore labels
        relabel_dict = dict((c, c) for c in np.unique(segmentation_mask))

        if self.ignore_labels is not None:
            # If ignore_labels label is not None, then we must remove it from the segmentation mask
            # We can do this by setting all pixels with values from `ignore_labels` to label 0
            segmentation_mask[np.isin(segmentation_mask, self.ignore_labels)] = 0
        else:
            # In case the ignore_labels is None, we must check whether the segmentation mask contains 0 values.
            # And if so - we must relabel the segmentation mask to ensure there are no 0 values,
            # since they are ignored by regionprops
            zeros_mask = segmentation_mask == 0
            if zeros_mask.any():
                new_label = segmentation_mask.max(initial=0) + 1  # Find the free label
                # Type promition is necessary if we have uint8 (for instance) mask with values already covering all range [0..255]
                # So next label will be 256, which is out of range for uint8. Therefore we must promote the type to uint16
                # But if the mask is already uint16, then we don't want to promote it to uint32
                # So min_scalar_type comes to the rescue of finding the minimal type that can hold the value of a new label.
                promote_type = np.promote_types(segmentation_mask.dtype, np.min_scalar_type(new_label))
                segmentation_mask = segmentation_mask.astype(promote_type, copy=False)
                segmentation_mask[zeros_mask] = new_label
                # We also have to update the relabel_dict to reflect the change of classes
                relabel_dict[new_label] = 0

        features = {
            SegmentationMaskFeatures.SegmentationMaskId: [],
            SegmentationMaskFeatures.SegmentationMaskLabel: [],
            SegmentationMaskFeatures.SegmentationMaskArea: [],
            SegmentationMaskFeatures.SegmentationMaskBoundingBoxArea: [],
            SegmentationMaskFeatures.SegmentationMaskPerimeter: [],
            SegmentationMaskFeatures.SegmentationMaskCenterOfMassX: [],
            SegmentationMaskFeatures.SegmentationMaskCenterOfMassY: [],
            SegmentationMaskFeatures.SegmentationMaskSolidity: [],
            SegmentationMaskFeatures.SegmentationMaskSparseness: [],
            SegmentationMaskFeatures.SegmentationMaskBoundingBoxWidth: [],
            SegmentationMaskFeatures.SegmentationMaskBoundingBoxHeight: [],
            SegmentationMaskFeatures.SegmentationMaskMorphologyOpeningArea: [],
            SegmentationMaskFeatures.SegmentationMaskMorphologyClosingArea: [],
            SegmentationMaskFeatures.SegmentationMaskMorphologyOpeningAreaRatio: [],
            SegmentationMaskFeatures.SegmentationMaskMorphologyClosingAreaRatio: [],
        }

        if shared_keys is not None:
            for key in shared_keys:
                features[key] = []

        regions = regionprops(segmentation_mask, cache=False)
        for region_id, region in enumerate(regions):
            features[SegmentationMaskFeatures.SegmentationMaskId].append(region_id)
            features[SegmentationMaskFeatures.SegmentationMaskLabel].append(relabel_dict[region.label])
            features[SegmentationMaskFeatures.SegmentationMaskArea].append(region.area)
            features[SegmentationMaskFeatures.SegmentationMaskBoundingBoxArea].append(region.bbox_area)
            features[SegmentationMaskFeatures.SegmentationMaskPerimeter].append(region.perimeter)
            features[SegmentationMaskFeatures.SegmentationMaskCenterOfMassX].append(region.centroid[0])
            features[SegmentationMaskFeatures.SegmentationMaskCenterOfMassY].append(region.centroid[1])
            features[SegmentationMaskFeatures.SegmentationMaskSolidity].append(region.solidity)
            features[SegmentationMaskFeatures.SegmentationMaskSparseness].append(region.area / region.area_filled)

            features[SegmentationMaskFeatures.SegmentationMaskBoundingBoxWidth].append(region.bbox[2] - region.bbox[0])
            features[SegmentationMaskFeatures.SegmentationMaskBoundingBoxHeight].append(region.bbox[3] - region.bbox[1])

            img_after_opening = binary_dilation(binary_erosion(region.image))
            img_after_closing = binary_erosion(binary_dilation(region.image))

            area_after_open = np.count_nonzero(img_after_opening)
            area_after_close = np.count_nonzero(img_after_closing)

            area_ratio_open = area_after_open / region.area
            area_ratio_close = area_after_close / region.area

            features[SegmentationMaskFeatures.SegmentationMaskMorphologyOpeningArea].append(area_ratio_open)
            features[SegmentationMaskFeatures.SegmentationMaskMorphologyClosingArea].append(area_ratio_close)

            features[SegmentationMaskFeatures.SegmentationMaskMorphologyOpeningAreaRatio].append(area_ratio_open)
            features[SegmentationMaskFeatures.SegmentationMaskMorphologyClosingAreaRatio].append(area_ratio_close)

            if shared_keys is not None:
                for key, value in shared_keys.items():
                    features[key].append(value)

        return features
