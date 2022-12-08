import torch


def squeeze_by_classes(label, is_one_hot) -> torch.Tensor:

    # Take all classes but ignored/background
    all_classes = torch.unique(label)[1:]

    # If no classes appear in the annotation it's a background image full of zeros
    if not all_classes.nelement():
        # TODO If it's one hot I should handle it differently
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
            mask = torch.where((label == cls) & (label > 0), torch.tensor(cls), torch.tensor(0)).squeeze()
            masks.append(mask)
    one_hot = torch.stack(masks, dim=0)
    return one_hot

