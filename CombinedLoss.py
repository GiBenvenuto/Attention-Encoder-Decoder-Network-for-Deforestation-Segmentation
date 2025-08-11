import torch


class BinaryDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: Log-probabilidades (saída de LogSoftmax), forma (N, 2, H, W)
        targets: Rótulos binários, forma (N, H, W)
        """
        preds = torch.exp(preds)

        preds_foreground = preds[:, 1, :, :]  # Probabilidades da classe 1

        # Targets devem estar no formato binário (0 ou 1)
        targets = targets.float()

        intersection = torch.sum(preds_foreground * targets, dim=(1, 2))
        pred_sum = torch.sum(preds_foreground, dim=(1, 2))
        target_sum = torch.sum(targets, dim=(1, 2))

        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        """
        preds: Log-probabilidades (saída de LogSoftmax), forma (N, 2, H, W)
        targets: Rótulos binários, forma (N, H, W)
        """
        probs = torch.exp(preds)

        probs_pos = probs[:, 1, :, :]  # Probabilidade da classe 1 (foreground)

        targets = targets.float()

        pt = probs_pos * targets + (1 - probs_pos) * (1 - targets)

        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        return focal_loss.mean()


class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, gamma=2.0):
        """
        alpha: Peso para Dice Loss
        beta: Peso para Focal Loss
        smooth: Valor para Dice Loss
        gamma: Parâmetro de Focal Loss
        """
        super(CombinedLoss, self).__init__()
        self.dice_loss = BinaryDiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(gamma=gamma)
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets):
        """
        preds: Log-probabilidades (saída de LogSoftmax), forma (N, 2, H, W)
        targets: Rótulos binários, forma (N, H, W)
        """
        dice = self.dice_loss(preds, targets)
        focal = self.focal_loss(preds, targets)
        return self.alpha * dice + self.beta * focal
