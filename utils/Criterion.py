import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):

    def __init__(self, eps: float, ignore_index: int, reduction: str = "mean" or "sum"):

        assert 0.0 < eps <= 1.0
        assert reduction == "mean" or reduction == "sum"

        super(LabelSmoothingLoss, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred: torch.tensor, target: torch.tensor):

        # pred: (batch_size * tgt_length, vocab_size)
        # target: (batch_size * tgt_length)
        pred = torch.log_softmax(pred, dim=-1)

        # target: (batch_size * tgt_length, 1)
        target = target.unsqueeze(1)

        # nll_loss: (batch_size * tgt_length, 1)
        nll_loss = -pred.gather(dim=-1, index=target)

        # smooth_loss: (batch_size * tgt_length, 1)
        smooth_loss = -pred.sum(dim=-1, keepdim=True)

        if self.ignore_index is not None:

            non_pad_mask = target.ne(self.ignore_index)
            nll_loss = nll_loss[non_pad_mask]
            smooth_loss = smooth_loss[non_pad_mask]

        if self.reduction == "mean":
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()
        else:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        return (1.0 - self.eps) * nll_loss + self.eps * smooth_loss / pred.size(-1)
