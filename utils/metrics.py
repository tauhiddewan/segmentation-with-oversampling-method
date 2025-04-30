import torch
# from torchmetrics import Dice, JaccardIndex

def calculate_dice_score(preds, targets, device, model_name):
    if model_name!="unet":
        preds = torch.sigmoid(preds)
    preds = torch.where(preds > 0.5, 1.0, 0.0)
    targets = torch.where(targets > 0.5, 1.0, 0.0)
    preds = preds.view(preds.size(0), -1).to(device)
    targets = targets.view(targets.size(0), -1).to(device) 
    tp = torch.sum(torch.mul(preds, targets))
    fp = torch.sum(preds * (1-targets))
    fn = torch.sum((1-preds) * targets)
    denominator = (2*tp+fp+fn)
    return (2 * tp) / denominator if denominator != 0 else torch.tensor(0.0, device=device)

def calculate_iou_score(preds, targets, device, model_name):
    if model_name!="unet":
        preds = torch.sigmoid(preds)
    preds = torch.where(preds > 0.5, 1.0, 0.0)
    targets = torch.where(targets > 0.5, 1.0, 0.0)
    preds = preds.view(preds.size(0), -1).to(device)
    targets = targets.view(targets.size(0), -1).to(device)
    tp = torch.sum(torch.mul(preds, targets))
    fp = torch.sum(preds * (1-targets))
    fn = torch.sum((1-preds) * targets)
    denominator = (tp + fp + fn)
    return tp / denominator if denominator != 0 else torch.tensor(0.0, device=device)


# def calculate_dice_score_pt(preds, targets, device):
#     preds = torch.sigmoid(preds)
#     preds = (preds > 0.5).float().to(device)
#     targets = (targets > 0.5).float().to(device)
#     targets = targets.to(torch.int32)
#     scorer = Dice(average='micro', threshold=0.5).to(device)
#     dice_score = scorer(preds, targets)
#     return dice_score

# def calculate_iou_score_pt(preds, targets, device):
#     preds = torch.sigmoid(preds)
#     preds = (preds > 0.5).float().to(device)
#     targets = (targets > 0.5).float().to(device)
#     targets = targets.to(torch.int32)
#     scorer = JaccardIndex(task='binary', threshold=0.5).to(device)
#     iou_score = scorer(preds, targets)
#     return iou_score