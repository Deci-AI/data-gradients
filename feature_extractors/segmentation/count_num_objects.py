from utils.data_classes import BatchData
from feature_extractors.segmentation.segmentation_abstract import SegmentationFeatureExtractorAbstract
from logger.logger_utils import create_bar_plot


class CountNumObjects(SegmentationFeatureExtractorAbstract):
    def __init__(self):
        super().__init__()
        self._number_of_objects_per_image = dict()

    def execute(self, data: BatchData):
        for image_contours in data.contours:
            num_objects_in_image = sum([len(cls_contours) for cls_contours in image_contours])
            if num_objects_in_image in self._number_of_objects_per_image:
                self._number_of_objects_per_image[num_objects_in_image] += 1
            else:
                self._number_of_objects_per_image.update({num_objects_in_image: 1})

    def process(self, ax, train):

        create_bar_plot(ax, self._number_of_objects_per_image, range(len(self._number_of_objects_per_image)),
                        x_label="# Objects in image", y_label="# Of images", title="# Objects per image",
                        train=train, color=self.colors[int(train)])

        ax.grid(visible=True, axis='y')
