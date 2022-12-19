import torch


def squeeze_by_classes(label: torch.Tensor, is_one_hot: bool) -> torch.Tensor:
    """
    Method gets label with the shape of [BS, N, W, H] where N is either 1 or num_classes, if is_one_hot=True.
    param label: Tensor
    param is_one_hot: Determine if labels are one-hot shaped
    :return: Labels tensor shaped as [BS, VC, W, H] where VC is Valid Classes only - ignores are omitted.
    """
    # Take all classes but ignored/background
    all_classes = torch.unique(label)[1:]

    # If no classes appear in the annotation it's a background image full of zeros
    if not all_classes.nelement():
        # TODO If it's one hot I should handle it differently. Also background images?
        return label

    masks = []
    # TODO: Check that method works for one-hots
    if is_one_hot:
        # Iterate over channels
        for i in range(label.shape[0]):
            # If the channel exist in the annotations
            if i in all_classes:
                mask = label[i, ...]
                mask = torch.where((mask > 0), torch.tensor(i), torch.zeros_like(mask))
                masks.append(mask)

    else:
        for cls in all_classes:
            mask = torch.where((label == cls) & (label > 0),
                               cls.clone(),
                               torch.tensor(0, dtype=cls.dtype)
                               ).squeeze()
            masks.append(mask)

    classes_hot = torch.stack(masks, dim=0)

    return classes_hot
