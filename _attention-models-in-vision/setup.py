import torch
from torch import nn
from torchvision import transforms as t, datasets as ds
from torchvision import datasets as ds
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import Subset

from losses import LabelSmoothingCrossEntropyLoss, CosineLoss
from models import Baseline, SEBaseline, AABaseline, SASABaseline, ChannelAttnBaseline,\
    SimpleResNet, SimpleSEResNet, ResNet, SEResNet, SimpleAAResNet, AAResNet, SimpleSASAResNet
from utils.data import BalancedDataset
from utils.runners import set_image_classification_epoch_runner


def get_model(opts):
    """
    ----------------------------------------------------------------------------------------------------------------
    Test(id=Baseline.SGD.CosineAnnealingLR.CIFAR10.1000.512.01, loss=1.6240, mA=0.8682)
    ✡ Test(id=SEBaseline.SGD.CosineAnnealingLR.CIFAR10.1000.512.01, loss=1.6408, mA=0.8717) ✡
    ----------------------------------------------------------------------------------------------------------------
    ✡ Test(id=Baseline.SGD.OneCycleLR.CIFAR10.1000.512.01, loss=1.6121, mA=0.8697) ✡
    Test(id=SEBaseline.SGD.OneCycleLR.CIFAR10.1000.512.01, loss=1.6210, mA=0.8695)
    Test(id=AABaseline.SGD.OneCycleLR.CIFAR10.1000.2048.01, loss=1.6297, mA=0.8625)
    TODO Test(id=SASABaseline.AdamW.OneCycleLR.CIFAR10.300.128.001, loss=1.6930, mA=0.8438)
    ----------------------------------------------------------------------------------------------------------------
    Test(id=SimpleResNet56.SGD.CosineAnnealingWarmRestarts.CIFAR10.1000.256.01, loss=1.5356, mA=0.9205)
    ✡ Test(id=SimpleSEResNet56.SGD.CosineAnnealingWarmRestarts.CIFAR10.1000.256.01, loss=1.5866, mA=0.9238) ✡
    ----------------------------------------------------------------------------------------------------------------
    Test(id=SimpleResNet56.SGD.CosineAnnealingLR.CIFAR10.1000.256.01, loss=1.5238, mA=0.9273)
    ✡ Test(id=SimpleSEResNet56.SGD.CosineAnnealingLR.CIFAR10.1000.512.01, loss=1.5145, mA=0.9353) ✡
    Test(id=SimpleStdAAResNet56.SGD.CosineAnnealingLR.CIFAR10.1000.512.01, loss=1.5243, mA=0.9275)
    ----------------------------------------------------------------------------------------------------------------
    Test(id=SimpleResNet56.SGD.OneCycleLR.CIFAR10.1000.256.01, loss=1.5239, mA=0.9254)
    ✡ Test(id=SimpleSEResNet56.SGD.OneCycleLR.CIFAR10.1000.512.01, loss=1.5160, mA=0.9356) ✡
    Test(id=SimpleOrigAAResNet56.SGD.OneCycleLR.CIFAR10.1000.512.01, loss=1.5261, mA=0.9138)
    Test(id=SimpleStdAAResNet56.SGD.OneCycleLR.CIFAR10.1000.512.01, loss=1.5381, mA=0.9265)
    Test(id=SimpleSASAResNet56.AdamW.OneCycleLR.CIFAR10.300.256.01, loss=0.8838, mA=0.8457)
    ----------------------------------------------------------------------------------------------------------------
    ✡ Test(id=SimpleResNet110.SGD.CosineAnnealingLR.CIFAR10.1000.1024.01, loss=1.5177, mA=0.9336) ✡
    Test(id=SimpleSEResNet110.SGD.CosineAnnealingLR.CIFAR10.1000.1024.01, loss=1.5231, mA=0.9319)
    ----------------------------------------------------------------------------------------------------------------
    Test(id=ResNet50.SGD.CosineAnnealingLR.CIFAR10.1000.128.01, loss=1.5777, mA=0.9244)
    ✡ Test(id=ResNet50.AdamW.OneCycleLR.CIFAR10.300.128.005, loss=1.5237, mA=0.9395) ✡
    Test(id=SEResNet50.SGD.CosineAnnealingLR.CIFAR10.1000.128.01, loss=1.5243, mA=0.9156)
    Test(id=AAResNet50.SGD.OneCycleLR.CIFAR10.1000.256.01, loss=1.5223, mA=0.9072)
    ----------------------------------------------------------------------------------------------------------------
    """
    return {
        'Baseline': lambda: Baseline(opts),
        'SEBaseline': lambda: SEBaseline(opts),
        'AABaseline': lambda: AABaseline(opts),
        'SASABaseline': lambda: SASABaseline(stem=False),
        'SASAStemBaseline': lambda: SASABaseline(stem=True),
        'SimpleChannelAttnBaseline': lambda: ChannelAttnBaseline(simple=True, mode='none'),
        'ComplexChannelAttnBaseline': lambda: ChannelAttnBaseline(simple=False, mode='none'),
        'SkipSimpleChannelAttnBaseline': lambda: ChannelAttnBaseline(simple=True, mode='skip'),
        'SkipComplexChannelAttnBaseline': lambda: ChannelAttnBaseline(simple=False, mode='skip'),
        'ScaleSimpleChannelAttnBaseline': lambda: ChannelAttnBaseline(simple=True, mode='scale'),
        'ScaleComplexChannelAttnBaseline': lambda: ChannelAttnBaseline(simple=False, mode='scale'),
        'SimpleResNet56': lambda: SimpleResNet(n=9),
        'SimpleResNet110': lambda: SimpleResNet(n=18),
        'SimpleSEResNet56': lambda: SimpleSEResNet(n=9),
        'SimpleSEResNet110': lambda: SimpleSEResNet(n=18),
        'SimpleOrigAAResNet56': lambda: SimpleAAResNet(n=9, original=True),
        'SimpleStdAAResNet56': lambda: SimpleAAResNet(n=9, original=False),
        'SimpleSASAResNet56': lambda: SimpleSASAResNet(n=9, stem=False),
        'SimpleStemSASAResNet56': lambda: SimpleSASAResNet(n=9, stem=True),
        'ResNet50': lambda: ResNet(sizes=[3, 4, 6, 3]),
        'SEResNet50': lambda: SEResNet(sizes=[3, 4, 6, 3]),
        'AAResNet50': lambda: AAResNet(sizes=[3, 4, 6, 3]),
        'ResNet101': lambda: ResNet(sizes=[3, 4, 23, 3]),
        'SEResNet101': lambda: SEResNet(sizes=[3, 4, 23, 3]),
    }[opts.model_name]()


def get_optimizer(opts, params):
    return {
        # lr=0.005 and batch_size=1024
        'Adam': lambda: torch.optim.Adam(params, lr=opts.lr),
        # lr=0.005 and batch_size=1024
        # lr=0.05 and batch_size=256
        'AdamW': lambda: torch.optim.AdamW(params, lr=opts.lr, weight_decay=0.1),
        # lr=0.1 and batch_size=512
        'SGD': lambda: torch.optim.SGD(params, lr=opts.lr, momentum=0.9, weight_decay=5e-4)
    }[opts.optimizer_name]()


def get_scheduler(opts, optimizer):
    return {
        'OneCycleLR': lambda: OneCycleLR(
            optimizer, max_lr=opts.lr, total_steps=opts.epochs),
        'CosineAnnealingLR': lambda: CosineAnnealingLR(
            optimizer, T_max=opts.epochs, eta_min=0, last_epoch=opts.last_epoch),
        'CosineAnnealingWarmRestarts': lambda: CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0, last_epoch=opts.last_epoch)
    }[opts.scheduler_name]()


def get_transforms(dataset, mode):
    """
    Get transforms for {dataset}.{mode}.
    Uses PyTorch name standards, or custom names.
    """
    transforms = {
        'CIFAR10.train': lambda: t.Compose([
            t.RandomCrop(32, padding=4),
            t.RandomHorizontalFlip(),
            t.ToTensor(),
            t.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'CIFAR10.validation': lambda: t.Compose([
            t.ToTensor(),
            t.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'CIFAR10.test': lambda: t.Compose([
            t.ToTensor(),
            t.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
    }
    transforms['small.CIFAR10.train'] = transforms['CIFAR10.train']
    transforms['small.CIFAR10.validation'] = transforms['CIFAR10.validation']
    transforms['small.CIFAR10.test'] = transforms['CIFAR10.test']
    return transforms[f'{dataset}.{mode}']()


def get_dataloader(opts, mode):
    """
    Get transforms for {dataset}.{mode}.
    Uses a lazy definition in the dictionary, then calls at the end.
    Uses PyTorch name standards, or custom names.
    """
    is_train = mode == 'train'
    data = {
        'CIFAR10.train': lambda: Subset(
            ds.CIFAR10(
                root='tmp/datasets', train=True, download=True,
                transform=get_transforms(opts.dataset_name, mode)
            ), range(5000, 50000)),
        'CIFAR10.validation': lambda: Subset(
            ds.CIFAR10(
                root='tmp/datasets', train=True, download=True,
                transform=get_transforms(opts.dataset_name, mode)
            ), range(5000)),
        'CIFAR10.test': lambda: ds.CIFAR10(
            root='tmp/datasets', train=False, download=True,
            transform=get_transforms(opts.dataset_name, mode)),
        'small.CIFAR10.train': lambda: BalancedDataset(Subset(
            ds.CIFAR10(
                root='tmp/datasets', train=True, download=True,
                transform=get_transforms(opts.dataset_name, mode)
            ), range(5000, 50000)
        ), max_per_class=50)
    }
    data['small.CIFAR10.validation'] = data['CIFAR10.validation']
    data['small.CIFAR10.test'] = data['CIFAR10.test']
    data = data[f'{opts.dataset_name}.{mode}']()
    return torch.utils.data.DataLoader(
        data, batch_size=opts.batch_size, shuffle=is_train,
        num_workers=opts.n_threads, pin_memory=True)


def get_epoch_runner_setter(opts):
    return {
        'image_classification': lambda: set_image_classification_epoch_runner
    }[opts.task]()


def get_criterion(opts):
    return {
        'CrossEntropyLoss': lambda: nn.CrossEntropyLoss(reduction='none'),
        'LabelSmoothingCrossEntropyLoss': lambda: LabelSmoothingCrossEntropyLoss(smoothing=0.1, reduction='none'),
        'BCELoss': lambda: nn.BCELoss(reduction='none'),
        'BCEWithLogitsLoss': lambda: nn.BCEWithLogitsLoss(pos_weight=opts.dataset_stats.class_weights, reduction='none'),
        'CosineLoss': lambda: CosineLoss(one_hot=False, reduction='none')
    }[opts.criterion]()
