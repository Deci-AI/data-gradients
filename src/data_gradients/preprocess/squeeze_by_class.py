from typing import List

import torch


def squeeze_by_classes(label: torch.Tensor, is_one_hot: bool, ignore_labels: List) -> torch.Tensor:
    """
    Method gets label with the shape of [BS, N, W, H] where N is either 1 or num_classes, if is_one_hot=True.
    param label: Tensor
    param is_one_hot: Determine if labels are one-hot shaped
    :return: Labels tensor shaped as [BS, VC, W, H] where VC is Valid Classes only - ignores are omitted.
    """
    if is_one_hot:
        all_classes = [i for i in range(label.shape[0]) if i not in ignore_labels]
    else:
        # Take all classes but ignored/background
        all_classes = [int(u.item()) for u in torch.unique(label) if u not in ignore_labels]

    # If no classes appear in the annotation it's a background image full of zeros
    if not all_classes:
        return label

    masks = []
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
                               torch.tensor(cls),
                               torch.tensor(0, dtype=torch.tensor(cls).dtype)
                               ).squeeze()
            masks.append(mask)

    classes_hot = torch.stack(masks, dim=0)

    return classes_hot
