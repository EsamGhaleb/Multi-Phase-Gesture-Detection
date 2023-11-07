import torch

def iou_score(preds, targets):
    """
    Compute the Intersection over Union (IoU) score between two sequences.

    :param preds: tensor of predictions of shape [batch size, seq_len]
    :param targets: tensor of targets of shape [batch size, seq_len]
    :return: mean IoU score for the batch
    """
    # calculate intersection and union for each sequence in the batch
    intersection = (preds == targets).sum(dim=1).float()
    union = torch.tensor([torch.numel(targets[0])]).float()
    iou = (intersection / union).mean()
    return iou
