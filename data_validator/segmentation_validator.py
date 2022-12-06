from data_validator.validator_abstract import ValidatorAbstract


class SegmentationValidator(ValidatorAbstract):
    def __init__(self):
        super().__init__()

    def validate(self, dataloader):
        images, labels = next(iter(dataloader))
        if images.dim() != 4:
            raise ValueError(f"Images batch shape should be (BatchSize x Channels x Width x Height). Got {images.shape}")
        if labels.dim() != 4:
            raise ValueError(f"Labels batch shape should be (BatchSize x Channels x Width x Height). Got {labels.shape}")
        if images[0].shape[0] != self._number_of_channels and images[0].shape[-1] != self._number_of_channels:
            raise ValueError(f"Images should have {self._number_of_channels} number of channels. Got {min(images[0].shape)}")
        self._images_channels_last = images[0].shape[0] != self._number_of_channels
        image_shape = images[0].shape[:2] if self._images_channels_last else images[0].shape[1:]
        labels_shape = labels[0].shape
        n_classes = min(labels[0].shape)
        if labels_shape == (1, *image_shape):
            # TODO: Check for binary
            # TODO: Check for values range
            # TODO: Think of how to save current format
            # TODO: Build reformats if needed
            pass
        elif labels_shape == (*image_shape, 1):
            pass
        elif labels_shape == (n_classes, *image_shape):
            pass
        elif labels_shape == (*image_shape, n_classes):
            pass
