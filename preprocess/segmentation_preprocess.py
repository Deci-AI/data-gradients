from batch_data import BatchData
from preprocess.preprocessor_abstract import PreprocessorAbstract


class SegmentationPreprocessor(PreprocessorAbstract):
    def __init__(self):
        super().__init__()

    def preprocess(self, images, labels) -> BatchData:
        # TODO preprocess inside manager doing segmentation thing!
        onehot_labels = [onehot.get_onehot(label) for label in labels]

        if debug_mode:
            for label, image in zip(onehot_labels, images):
                temp = contours.get_contours(label, image)
                break

        onehot_contours = [contours.get_contours(onehot_label) for onehot_label in onehot_labels]

        if self._images_channels_last:
            # TODO: Check that works
            images = images.permute(0, -1, 1, 2)

        bd = BatchData(images=images,
                       labels=labels,
                       batch_onehot_contours=onehot_contours,
                       batch_onehot_labels=onehot_labels)

        return bd