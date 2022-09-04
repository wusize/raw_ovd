import torch


def centerness_score(preds, gts):
    # xyxy
    pred_centers = preds.view(-1, 2, 2).mean(1)
    ls = (pred_centers[:, 0] - gts[:, 0]).clamp(min=0.0)
    rs = (gts[:, 2] - pred_centers[:, 0]).clamp(min=0.0)
    us = (pred_centers[:, 1] - gts[:, 1]).clamp(min=0.0)
    ds = (gts[:, 3] - pred_centers[:, 1]).clamp(min=0.0)

    lrs = torch.stack([ls, rs], dim=-1)
    uds = torch.stack([us, ds], dim=-1)

    return torch.sqrt(lrs.min(-1).values * uds.min(-1).values
                      / (lrs.max(-1).values * uds.max(-1).values + 1e-12)
                      )
