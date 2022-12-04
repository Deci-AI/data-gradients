import torch


def get_onehot(label) -> torch.Tensor:
    all_classes = torch.unique(label)[1:]
    masks = []
    for cls in all_classes:
        mask = torch.where((label == cls) & (label > 0), torch.tensor(cls), torch.tensor(0))
        masks.append(mask)

    one_hot = torch.concat(masks, dim=0)
    return one_hot
