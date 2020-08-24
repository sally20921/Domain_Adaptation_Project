from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler

from transformations.simclr import TransformsSimCLR


class Datasets:

    @staticmethod
    def get_simclr_dataset(config):
        train_dataset = torchvision.datasets.CIFAR10(
                config.simclr.train.data_dir_path,
                train=True,
                download=True,
                transform=TransformsSimCLR(size=config.simclr.train.img_size)
        )

        classes = train_dataset.classes

        return train_dataset, classes

    @staticmethod
    def get_datasets(config, img_size=None):

        if img_size is None:
            img_size = config.simclr.train.img_size

        
        train_dataset = torchvision.datasets.CIFAR10(
                config.simclr.train.data_dir_path,
                train=True,
                download=True,
                transform=TransformsSimCLR(size=img_size).train_test_transform,
        )

        val_dataset = torchvision.datasets.CIFAR10(
                config.simclr.train.data_dir_path,
                train=True,
                download=True,
                transform=TransformsSimCLR(size=img_size).test_transform,
        )

        classes = train_dataset.classes

        test_dataset = torchvision.datasets.CIFAR10(
                config.simclr.train.data_dir_path,
                train=False,
                download=True,
                transform=TransformsSimCLR(size=img_size).test_transform,
        )

        return train_dataset, val_dataset, test_dataset, classes

    @staticmethod
    def get_simclr_loader(config, train_dataset):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.simclr.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.simclr.train.num_workers,
            sampler=None,
        )

        return train_loader

    @staticmethod
    def get_loaders(config, train_dataset, val_dataset, test_dataset):

        # no separate val set so the val images have to be samples from the same dataset
        if config.simclr.train.dataset == 'CIFAR10':

            valid_size = 0.2
            num_train = len(train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(valid_size * num_train))

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config.simclr.train.batch_size, sampler=train_sampler,
                num_workers=config.simclr.train.num_workers,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=config.simclr.train.batch_size, sampler=valid_sampler,
                num_workers=config.simclr.train.num_workers,
            )
        else:

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.simclr.train.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=config.simclr.train.num_workers,
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config.simclr.train.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=config.simclr.train.num_workers,
            )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.simclr.train.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=config.simclr.train.num_workers,
        )

        return train_loader, val_loader, test_loader
