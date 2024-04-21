import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torchvision import models
import pytorch_lightning as pl

from utils import conv_block, fscore


class CustomVgg16(pl.LightningModule):
    def __init__(self, num_classes=10, pretrained=True, learning_rate=1e-3):
        super().__init__()
        self.model = models.vgg16(pretrained=pretrained)
        self.num_classes = num_classes
        # freeze layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # update final linear layer
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

        # other parameters
        self.criterion = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        # labels = nn.functional.one_hot(labels, num_classes=self.num_classes).sum(dim=0).float()
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        # labels = nn.functional.one_hot(labels, num_classes=self.num_classes).sum(dim=0).float()
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True),
            'monitor': 'val_loss'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class SimpleResNet(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super().__init__()
        # other parameters
        self.criterion = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = 1e-4

        self.save_hyperparameters()
        self.conv1 = conv_block(3, 64)  # 64, 128, 128
        self.res1 = nn.Sequential(
            conv_block(64, 64),
            conv_block(64, 64)
        )  # 64, 128, 128

        self.conv2 = conv_block(64, 128, pool=True)  # output 128 x 32 x 32
        self.res2 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128),
            conv_block(128, 128)
        )  # output 128 x 32 x 32

        self.conv3 = conv_block(128, 512, pool=True)  # output 512 x 8 x 8
        self.res3 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )  # output 512 x 8 x 8

        self.conv4 = conv_block(512, 1024, pool=True)  # output 1024 x 2 x 2
        self.res4 = nn.Sequential(
            conv_block(1024, 1024),
            conv_block(1024, 1024)
        )  # output 1024 x 2 x 2

        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),  # output 1024 x 1 x 1
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1024 * 1 * 1, 512),  # output 512
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )  # output 10

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out) + out
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.conv4(out)
        out = self.res4(out) + out
        out = self.classifier(out)
        # out = F.sigmoid(out)

        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        # labels = nn.functional.one_hot(labels, num_classes=self.num_classes).sum(dim=0).float()
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        # labels = nn.functional.one_hot(labels, num_classes=self.num_classes).sum(dim=0).float()
        loss = self.criterion(outputs, labels)
        acc = fscore(outputs, labels)
        self.log_dict({'val_loss': loss, 'val_score': acc})
        return {'val_loss': loss, 'val_score': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True),
            'monitor': 'val_loss'
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
