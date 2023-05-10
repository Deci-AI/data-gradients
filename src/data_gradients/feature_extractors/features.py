from enum import Enum


class ImageFeatures(Enum):
    """
    Define the features that are extracted from the image itself.

    DatasetSplit: str - The name of the dataset split. Could be "train", "val", "test", etc.
    ImageId: str - The unique identifier of the image. Could be the image path or the image name.
                   It is not that relevant what exactly this string is, as long as it is unique per dataset.
    ImageWidth: int - The width of the image in pixels.
    ImageHeight: int - The height of the image in pixels.
    ImageArea: int - The area of the image in pixels.
    ImageAspectRatio: float - The aspect ratio of the image. Calculated as width/height.
    ImageNumChannels: int - The number of channels in the image.

    ImageMean: float[] - The mean value of the image pixels calculated per channel.
    ImageStd: float[] - The standard deviation of the image pixels calculated per channel.

    ImageAvgBrightness: float - The average brightness of the image pixels. To calculate this, the image is converted to grayscale.
    ImageMinBrightness: float - The minimum brightness of the image pixels. To calculate this, the image is converted to grayscale.
    ImageMaxBrightness: float - The maximum brightness of the image pixels. To calculate this, the image is converted to grayscale.

    """
    DatasetSplit = "DatasetSplit"
    ImageId = "ImageId"

    ImageWidth = "ImageWidth"
    ImageHeight = "ImageHeight"
    ImageArea = "ImageArea"
    ImageAspectRatio = "ImageAspectRatio"
    ImageNumChannels = "ImageNumChannels"

    ImageMean = "ImageMean"
    ImageStd = "ImageStd"

    ImageAvgBrightness = "ImageAvgBrightness"
    ImageMinBrightness = "ImageMinBrightness"
    ImageMaxBrightness = "ImageMaxBrightness"


class BoundingBoxFeatures(Enum):
    """
    Define the features that are extracted from the bounding boxes.

    DatasetSplit: str - The name of the dataset split. Could be "train", "val", "test", etc.
    ImageId: str - The unique identifier of the image. Could be the image path or the image name.

    BoundingBoxId: str - The unique identifier of the bounding box. Could be the bounding box index in the image or a unique string.
    BoundingBoxLabel: str - The label of the bounding box. Could be the class name or the class id.
    BoundingBoxIsCrowd: bool - Whether the bounding box is a crowd bounding box or not.

    BoundingBoxWidth: int - The width of the bounding box in pixels.
    BoundingBoxHeight: int - The height of the bounding box in pixels.
    BoundingBoxAspectRatio: float - The aspect ratio of the bounding box. Calculated as width/height.
    BoundingBoxArea: int - The area of the bounding box in pixels.
    BoundingBoxAreaOutsideImage: int - The area of the bounding box that is outside the image in pixels.
    BoundingBoxCenterX: float - The x coordinate of the center of the bounding box in pixels.
    BoundingBoxCenterY: float - The y coordinate of the center of the bounding box in pixels.

    BoundingBoxMaxOverlap: float - The maximum overlap of the bounding box with any other bounding box in the image.

    """
    DatasetSplit = "DatasetSplit"
    ImageId = "ImageId"

    BoundingBoxId = "BoundingBoxId"
    BoundingBoxLabel = "BoundingBoxLabel"
    BoundingBoxIsCrowd = "BoundingBoxIsCrowd"

    BoundingBoxWidth = "BoundingBoxWidth"
    BoundingBoxHeight = "BoundingBoxHeight"
    BoundingBoxAspectRatio = "BoundingBoxAspectRatio"
    BoundingBoxArea = "BoundingBoxArea"
    BoundingBoxAreaOutsideImage = "BoundingBoxAreaOutsideImage"
    BoundingBoxCenterX = "BoundingBoxCenterX"
    BoundingBoxCenterY = "BoundingBoxCenterY"

    BoundingBoxMaxOverlap = "BoundingBoxMaxOverlap"


class SegmentationMaskFeatures(Enum):
    """
    Define the features that are extracted from the segmentation masks. Each row represents a single mask instance.

    DatasetSplit: str - The name of the dataset split. Could be "train", "val", "test", etc.
    ImageId: str - The unique identifier of the image. Could be the image path or the image name.

    SegmentationMaskId: str - The unique identifier of the segmentation mask within the image.
    SegmentationMaskLabel: str - The class label of the instance.

    SegmentationMaskArea: int - The area of the segmentation mask in pixels.
    SegmentationMaskPerimeter: int - The perimeter of the segmentation mask in pixels.
    SegmentationMaskCenterOfMassX: float - The x coordinate of the center of mass of the segmentation mask in pixels.
    SegmentationMaskCenterOfMassY: float - The y coordinate of the center of mass of the segmentation mask in pixels.
    SegmentationMaskSolidity: float - The solidity of the segmentation mask. Calculated as area/convex_area.
    SegmentationMaskSparseness: float - The sparseness of the segmentation mask. Calculated as area/area filled.
    SegmentationMaskBoundingBoxArea: int - The area of the bounding box of the segmentation mask in pixels.

    SegmentationMaskBoundingBoxWidth: int - The width of the corresponding bounding box for given instance mask in pixels.
    SegmentationMaskBoundingBoxHeight: int - The height of the corresponding bounding box for given instance mask in pixels.

    SegmentationMaskMorphologyOpeningArea: int - The area of the segmentation mask after applying a morphological
                                                 opening operation in pixels.

    SegmentationMaskMorphologyOpeningAreaRatio: float - The ratio of the area of the segmentation mask after
                                                        applying a morphological opening operation to the original
                                                        area of the segmentation mask.

    SegmentationMaskMorphologyClosingArea: int - The area of the segmentation mask after applying a morphological
                                                 closing operation in pixels.

    SegmentationMaskMorphologyClosingAreaRatio: float - The ratio of the area of the segmentation mask after
                                                        applying a morphological closing operation to the original
                                                        area of the segmentation mask.

    """

    DatasetSplit = "DatasetSplit"
    ImageId = "ImageId"

    SegmentationMaskId = "SegmentationMaskId"
    SegmentationMaskLabel = "SegmentationMaskLabel"

    SegmentationMaskArea = "SegmentationMaskArea"
    SegmentationMaskPerimeter = "SegmentationMaskPerimeter"
    SegmentationMaskCenterOfMassX = "SegmentationMaskCenterOfMassX"
    SegmentationMaskCenterOfMassY = "SegmentationMaskCenterOfMassY"
    SegmentationMaskSolidity = "SegmentationMaskSolidity"
    SegmentationMaskSparseness = "SegmentationMaskSparseness"
    SegmentationMaskBoundingBoxArea = "SegmentationMaskBoundingBoxArea"

    SegmentationMaskBoundingBoxWidth = "SegmentationMaskBoundingBoxWidth"
    SegmentationMaskBoundingBoxHeight = "SegmentationMaskBoundingBoxHeight"

    SegmentationMaskMorphologyOpeningArea = "SegmentationMaskMorphologyOpeningArea"
    SegmentationMaskMorphologyOpeningAreaRatio = "SegmentationMaskMorphologyOpeningAreaRatio"

    SegmentationMaskMorphologyClosingArea = "SegmentationMaskMorphologyClosingArea"
    SegmentationMaskMorphologyClosingAreaRatio = "SegmentationMaskMorphologyClosingAreaRatio"
