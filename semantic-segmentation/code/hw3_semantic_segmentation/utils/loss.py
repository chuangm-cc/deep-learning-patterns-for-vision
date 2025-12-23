import torch.nn as nn
import torch.nn.functional as F
import torch 

# =========================== Define your custom loss function ===========================================
class MyLoss(nn.Module):
    def __init__(self,device):
      super(MyLoss, self).__init__()
      self.device = device

    def forward(self, predictions, targets):
        # print(predictions.shape)
        # print(targets.shape)
        # do softmax
        # predictions= torch.softmax(predictions, dim=1)  # Assuming the second dimension is the class dimension
        # # print(predictions.shape)
        # # print(targets_onehot.shape)
        # predictions = predictions[:, 1, :, :]
        # intersection = torch.sum(predictions * targets)
        # union = torch.sum() + torch.sum(targets) + 1.

        # dice_loss = 1 - (2 * intersection + 1.) / union
        # ce_loss = F.smooth_l1_loss(
        #     predictions, targets)
        alpha = 1.
        beta = 1.
        ce_loss = F.cross_entropy(
            predictions, targets,ignore_index=255, reduction='none')
        exp_ce_loss = alpha*torch.exp(-ce_loss * beta)
        loss = ce_loss*exp_ce_loss
        return ce_loss.mean()

        # return dice_loss