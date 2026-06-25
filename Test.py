import torch
import matplotlib.pyplot as plt
from DataLoader import *
from Metrics import *
from Model.LiteHAM-Net import *
model = LiteHAM_Net()
model.eval()
# Dataset & Data Loader
import argparse


parser = argparse.ArgumentParser(description="Train a segmentation model.")
parser.add_argument("--path_checkpoint", type=str, default="./checkpoints", help="Path to save checkpoints")
parser.add_argument("--path_image_test", type=str, required=True, help="Path to testing images .npy file")
parser.add_argument("--path_label_test", type=str, required=True, help="Path to testing labels .npy file")
args = parser.parse_args()
CHECKPOINT_PATH = sorted(args.path_checkpoint)[0]
# Prediction

x_test = np.load(args.path_image_test)
y_test = np.load(args.path_label_test)
test_dataset = DataLoader(PH2Loader(x_test, y_test, typeData="test"), batch_size=1, num_workers=2, prefetch_factor=16)


def dice_score(pred, target, smooth=1e-5):
    """
    Compute Dice score for binary segmentation.

    Args:
    - pred: The predicted mask.
    - target: The ground truth mask.
    - smooth: A small smoothing constant to prevent division by zero.

    Returns:
    - Dice score as a float.
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Segmentor(pl.LightningModule):
    def __init__(self, model=model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred, decoder = self.model(image)
        loss = DiceLoss()(y_pred, y_true)
        print(loss.cpu().numpy(), end = ' ')
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        metrics = {"Test Dice": dice, "Test Iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics

trainer = pl.Trainer()

segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model = model, map_location = device)

trainer.test(segmentor, test_dataset)


