import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from datasets import load_dataset

from model import CustomVgg16, SimpleResNet


transform_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
transform_small = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
def prepare_data_old(data_dir='data/', batch_size=32, val_split=0.1):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
    test_set = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)

    # split train further into train and val
    split_index = int((1-val_split) * len(train_set))
    train_idx, val_idx = list(range(len(train_set)))[:split_index], list(range(len(train_set)))[split_index:]

    train_sampler = Subset(train_set, train_idx)
    val_sampler = Subset(train_set, val_idx)
    train_loader = DataLoader(train_sampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sampler, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def transform_hf(examples):
    examples['image'] = [transform_small(image) for image in examples['image']]
    # examples['label'] = [lbl['label'] for lbl in examples['digits']]
    return examples


def collate_fn(examples):
    images = []
    labels = torch.zeros((len(examples), 10), dtype=torch.float32)
    for i, sample in enumerate(examples):
        images.append(transform_small(sample['image']))
        for j in sample['digits']['label']:
            labels[i][j] = 1.

    return {'image': torch.stack(images), 'label': labels}


def load_data_hf(data_dir='data/', batch_size=32, val_split=0.1):
    dataset = load_dataset('svhn', 'full_numbers')
    train_set = dataset['train']  # .with_format('torch')
    val_set = dataset['test']  # .with_format('torch')

    # train_set = train_set.with_transform(transform_hf)
    # val_set = val_set.with_transform(transform_hf)

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader

if __name__ == '__main__':
    # train_loader, val_loader, test_loader = prepare_data()
    train_loader, val_loader = load_data_hf()
    model = CustomVgg16()
    # model = SimpleResNet()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='resnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=[checkpoint_callback],
        accelerator='auto'
    )
    trainer.fit(model, train_loader, val_loader)
