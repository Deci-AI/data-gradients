# Dataset Adapters

Datasets in computer vision are often stored in a variety of formats. 
There is no single "one size fits all" format that is best for all applications. 
Therefore, providing support for all the different formats is important but challenging at the same time.

Data Gradients solve this problem by introducing a concepts of a `DatasetSample` and `DatasetAdapter`.
Instead of having to support dozens of dataset formats we ask users to implement a single `DatasetAdapter` for a dataset they are working with.
For convinience, we provide a number of `DatasetAdapters` for popular datasets already. So you don't have to implement them yourself.
If you don't see an adapter for a dataset you are working with, please consider opening an issue or contributing one.

Data Gradients support following types of computer vision tasks:

- Semantic Segmentation
- Object Detection (In progress)


## Semantic Segmentation

Semantic segmentation takes an image as input and outputs a semantic segmentation mask of the same size.
In the nutshell this is a dense classification task where each pixel of the input image is assigned into one of the classes.

We provide a `SegmentationSample` class that represents a sample of such dataset:

```python
@dataclasses.dataclass
class SegmentationSample:
    """
    This is a dataclass that represents a single sample of the dataset.
    Support of different types of dataset formats is achieved by using adapters that should return SegmentationSample.

    Properties:
        sample_id: str
        image: np.ndarray of shape [H,W,C]
        mask: np.ndarray of shape [H,W] with integer values representing class labels
    """

    sample_id: str
    image: np.ndarray
    mask: np.ndarray
```

This is the basic unit that Data Gradients can operate with. To provide a dataset-level insights we use interface `SegmentationDatasetAdapter` 
that all adapters for semantic segmentation datasets must implement. 

It provides required metadata about the dataset (A number of classes, class names and what classes are ignored).
The most important method of the `SegmentationDatasetAdapter` is `get_iterator` that returns an iterable object that allow to iterate over the dataset.

By contract, this method should return an iterator of `SegmentationSample` objects.

```python
class SegmentationDatasetAdapter(abc.ABC):
    """
    This is an abstract class that represents a dataset adapter.
    It acts as a bridge interface between any specific dataset/dataloader/raw data on disk and the analysis manager.
    """

    @abc.abstractmethod
    def get_num_classes(self) -> int:
        ...

    @abc.abstractmethod
    def get_class_names(self) -> List[str]:
        ...

    @abc.abstractmethod
    def get_ignored_classes(self) -> Optional[List[int]]:
        ...

    @abc.abstractmethod
    def get_iterator(self) -> Iterable[SegmentationSample]:
        """
        This method should return an iterable object that contains SegmentationSample objects.
        It could read samples from disk, existing dataset, dataloader, etc.
        """
        raise NotImplementedError()

    def __len__(self):
        return None
```

## Writing adapters for custom datasets

To write an adapter for a custom dataset you need to implement `SegmentationDatasetAdapter` interface. 
And it is totally fine to take an existing dataset implementation class from your training pipeline and wrap it into an adapter to use it with Data Gradients.

There are few caveats to keep in mind when writing a custom dataset adapter:

- Your dataset class may be doing image preprocessing (e.g. resize, crop, normalization, etc.) inside. Depending on the use-case this may or may not be desired. 
  Just keep this in mind - if the dataset is doing preprocessing (say resizing all images to 224x224) - some data insights would be useless.
  If this is undesirable, you may want to configure your dataset without preprocessing steps.
- Your dataset class may be doing data augmentation (e.g. random crop, random flip, etc.) inside. 
  Again, depending on your intention this may or may not be desired. For instance, you may want to compare how image augmentations affect the model performance. 
  So you can pass two datasets with and without augmentations to Data Gradients and compare the results.
- If your dataset returns torch tensors, dataset adapter should convert them to numpy arrays (Including channel reordering CHW -> HWC). 


## Working with data-loaders

Data-loaders can be also used with Data Gradients. For built-in dataset adapters, you simply call `TorchvisionCityscapesSegmentationAdapter.wrap_dataloader` method.
For custom dataset adapters, you still have to write an adapter for your dataset first, but the `get_iterator` method would be implemented slightly differently:

```python
class MyCustomDataLoaderAdapter(SegmentationDatasetAdapter):
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        
    def get_iterator(self) -> Iterable[SegmentationSample]:
        for batch in self.dataloader:
            for image, mask in zip(batch["image"], batch["mask"]):
                yield SegmentationSample(
                    sample_id=batch["sample_id"],
                    image=image.numpy(),
                    mask=mask.numpy(),
                )
```


## Supported datasets 

| Dataset Name             | Adapter Name                               | Task                  |
|--------------------------|--------------------------------------------|-----------------------|
| BDD100K                  | `BDD100KSegmentationDatasetAdapter`        | Semantic Segmentation |
| Cityscapes (raw)         | `CityscapesSegmentationDatasetAdapter`     | Semantic Segmentation |
| Cityscapes (torchvision) | `TorchvisionCityscapesSegmentationAdapter` | Semantic Segmentation |
