import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import pytorch_lightning as pl


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
