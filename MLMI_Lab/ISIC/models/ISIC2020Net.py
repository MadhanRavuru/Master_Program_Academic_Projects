import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_lightning import Callback
from torchvision.models import resnet18, densenet121
from efficientnet_pytorch import EfficientNet
import pandas as pd
from glob import glob
from preprocessing.ISIC2020Transforms import get_train_transforms, get_valid_transforms, get_tta_transforms
from preprocessing.ISIC2020_alt import ImageDataset
from pytorch_lightning.metrics.classification import AUROC

seed = 42


class ISIC2020Net(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        '''
        self.net = resnet18(pretrained=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1, bias=True)
        '''
        self.net = densenet121(pretrained=True)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, 1, bias = True)
        
        self.dataset = {}

    def prepare_data(self) -> None:
        DATA_PATH = 'compressedISIC'
        TRAIN_ROOT_PATH = f'{DATA_PATH}/512x512-dataset-melanoma/512x512-dataset-melanoma'
        TEST_ROOT_PATH = f'{DATA_PATH}/512x512-test/512x512-test'

        df_folds = pd.read_csv(f'{DATA_PATH}/folds.csv', index_col='image_id',
                               usecols=['image_id', 'fold', 'target'], dtype={'fold': np.byte, 'target': np.byte})
        self.df_test = pd.read_csv(f'{DATA_PATH}/test.csv', index_col='image_name')

        df_folds = df_folds.sample(frac=1.0, random_state=seed * 6 + self.hparams["fold_number"])
        self.dataset["train"] = ImageDataset(
            path=TRAIN_ROOT_PATH,
            image_ids=df_folds[df_folds['fold'] != self.hparams["fold_number"]].index.values,
            labels=df_folds[df_folds['fold'] != self.hparams["fold_number"]].target.values,
            transforms=get_train_transforms(),
        )

        self.dataset["val"] = ImageDataset(
            path=TRAIN_ROOT_PATH,
            image_ids=df_folds[df_folds['fold'] == self.hparams["fold_number"]].index.values,
            labels=df_folds[df_folds['fold'] == self.hparams["fold_number"]].target.values,
            transforms=get_valid_transforms(),
        )
        
        self.dataset["test"] = ImageDataset(
            path=TEST_ROOT_PATH,
            image_ids=self.df_test.index.values,
            transforms=get_tta_transforms(),
        )

    @pl.data_loader
    def train_dataloader(self):
        train_dl = DataLoader(self.dataset["train"], batch_size=self.hparams["batch_size"],
                          num_workers=self.hparams["num_workers"],
                          drop_last=True, shuffle=True, pin_memory=True)
        print(len(train_dl.sampler))
        return train_dl

    @pl.data_loader
    def val_dataloader(self):
        val_dl = DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"],
                          num_workers=self.hparams["num_workers"],
                          drop_last=False, shuffle=False, pin_memory=True)
        return val_dl

    @pl.data_loader
    def test_dataloader(self):
        test_dl = DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"],
                          num_workers=self.hparams["num_workers"],
                          drop_last=False, shuffle=False, pin_memory=False)
        return test_dl

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["wd"])

        """
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            max_lr=self.hparams["lr"],
            optimizer=optimizer,
            base_momentum=0.90,
            max_momentum=0.95,
        )
        """
        return optimizer  # [optimizer], [scheduler]

    def step(self, batch):
        # return batch loss
        x, y = batch
        y_hat = self(x).flatten()
        # use label smoothing for regularization
        y_smo = y.float() * (1 - self.hparams["label_smoothing"]) + 0.5 * self.hparams["label_smoothing"]
        loss = F.binary_cross_entropy_with_logits(y_hat, y_smo.type_as(y_hat),
                                                  pos_weight=torch.tensor(self.hparams["pos_weight"]))
        return loss, y, y_hat.sigmoid()

    def training_step(self, batch, batch_nb):
        loss, y, y_hat = self.step(batch)
        acc = (y_hat.round() == y).float().mean().item()
        tensorboard_logs = {'train_loss': loss, 'acc': acc}
        return {'loss': loss, 'acc': acc, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, y, y_hat = self.step(batch)
        return {'val_loss': loss,
                'y': y.detach(), 'y_hat': y_hat.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        auc = AUROC()(pred=y_hat, target=y) if y.float().mean() > 0 else 0.5  # skip sanity check
        acc = (y_hat.round() == y).float().mean().item()
        print(f"Epoch {self.current_epoch} acc:{acc} auc:{auc}")
        tensorboard_logs = {'val_loss': avg_loss, 'val_auc': auc, 'val_acc': acc}
        return {'avg_val_loss': avg_loss,
                'val_auc': auc, 'val_acc': acc,
                'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, _ = batch
        y_hat = self(x).flatten().sigmoid()
        return {'y_hat': y_hat}

    def test_epoch_end(self, outputs):
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        self.df_test['target'] = y_hat.tolist()
        N = len(glob('submission*.csv'))
        self.df_test.target.to_csv(f'submission{N}.csv')
        return {'tta': N}
