import torch
from torch import nn
from torchvision import transforms as t, datasets as ds
from torchvision import datasets as ds
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import Subset

from data import AdultCensusDataset
from losses import LabelSmoothingLoss
from models import BaselinePredictor, EnsembleIndependentPredictor, MHAStackedEnsemble, NaiveStackedEnsemble,\
    MeanStackedEnsemble, MHAConvexCombinationStackedEnsemble, TransformerBasedStackedEnsemble,\
    NaiveConvexCombinationStackedEnsemble, TransformerBasedRandomizedStackedEnsemble, BaselineTransformer
from utils.runners import set_image_classification_epoch_runner, set_adult_census_epoch_runner


def get_model(opts):
    """
    ----------------------------------------------------------------------------------------------------------------
    Trained with OneCycleLR and AdamW.
    ----------------------------------------------------------------------------------------------------------------
    Test(model=BaselineAdultCensus, loss=0.5248, mA=0.8315, f1=0.6373)
    ✡ Test(model=BaselineTransformerAdultCensus, loss=0.5160, mA=0.8159, f1=0.6450)
    Test(model=TransformerBasedRandomizedStackedEnsembleAdultCensus, loss=0.5202, mA=0.8149, f1=0.6448)

    Test(model=EnsembleIndependent1AdultCensus, loss=0.5920, mA=0.7578, f1=0.5930)
    Test(model=EnsembleIndependent2AdultCensus, loss=0.6126, mA=0.7549, f1=0.5953)
    Test(model=EnsembleIndependent3AdultCensus, loss=0.6683, mA=0.7788, f1=0.4614)
    ----------------------------------------------------------------------------------------------------------------
    These use the EnsembleIndependent predictors reported above.
    ----------------------------------------------------------------------------------------------------------------
    Test(model=MeanEnsembleAdultCensus, loss=0.5606, mA=0.8164, f1=0.6117)
    Test(model=NaiveConvexCombinationStackedEnsembleAdultCensus, loss=0.5540, mA=0.8281, f1=0.6096)
    Test(model=MHAConvexCombinationStackedEnsembleAdultCensus, loss=0.5613, mA=0.8145, f1=0.6065)

    Test(model=NaiveStackedEnsembleAdultCensus, loss=0.5267, mA=0.8237, f1=0.6384)
    Test(model=MHAStackedEnsembleAdultCensus, loss=0.5288, mA=0.8145, f1=0.6362)
    ✡ Test(model=TransformerBasedStackedEnsembleAdultCensus, loss=0.5254, mA=0.8184, f1=0.6420)
    ----------------------------------------------------------------------------------------------------------------
    """
    return {
        'BaselineAdultCensus': lambda: BaselinePredictor(),
        'EnsembleIndependent1AdultCensus': lambda: EnsembleIndependentPredictor(
            features=(1, 3, 5), embeddings=(5, 8, 4)),
        'EnsembleIndependent2AdultCensus': lambda: EnsembleIndependentPredictor(
            features=(6, 7, 8, 9), embeddings=(8, 3, 3, 1)),
        'EnsembleIndependent3AdultCensus': lambda: EnsembleIndependentPredictor(
            features=(0, 2, 4, 10, 11, 12, 13), embeddings=(None, None, None, None, None, None, 21)),
        'MeanStackedEnsembleAdultCensus': lambda: MeanStackedEnsemble(),
        'NaiveConvexCombinationStackedEnsembleAdultCensus': lambda: NaiveConvexCombinationStackedEnsemble(),
        'NaiveStackedEnsembleAdultCensus': lambda: NaiveStackedEnsemble(),
        'MHAConvexCombinationStackedEnsembleAdultCensus': lambda: MHAConvexCombinationStackedEnsemble(),
        'MHAStackedEnsembleAdultCensus': lambda: MHAStackedEnsemble(),
        'TransformerBasedStackedEnsembleAdultCensus': lambda: TransformerBasedStackedEnsemble(),
        'TransformerBasedRandomizedStackedEnsembleAdultCensus': lambda: TransformerBasedRandomizedStackedEnsemble(),
        'BaselineTransformerAdultCensus': lambda: BaselineTransformer(),
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
            optimizer, max_lr=opts.lr, total_steps=opts.epochs, anneal_strategy='linear'),
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
    return {
    }[f'{dataset}.{mode}']()


def get_dataloader(opts, mode):
    """
    Get transforms for {dataset}.{mode}.
    Uses a lazy definition in the dictionary, then calls at the end.
    Uses PyTorch name standards, or custom names.
    """
    is_train = mode == 'train'
    data = {
        'AdultCensus.train': lambda: Subset(
            AdultCensusDataset(), range(0, 20000)),
        'AdultCensus.validation': lambda: Subset(
            AdultCensusDataset(), range(20000, 25000)),
        'AdultCensus.test': lambda: Subset(
            AdultCensusDataset(), range(25000, 32561))
    }[f'{opts.dataset_name}.{mode}']()
    return torch.utils.data.DataLoader(
        data, batch_size=opts.batch_size, shuffle=is_train,
        num_workers=opts.n_threads, pin_memory=True)


def get_epoch_runner_setter(opts):
    return {
        'image_classification': lambda: set_image_classification_epoch_runner,
        'adult_census': lambda: set_adult_census_epoch_runner
    }[opts.task]()


def get_criterion(opts):
    return {
        'CrossEntropyLoss': lambda: nn.CrossEntropyLoss(reduction='none'),
        'LabelSmoothing': lambda: LabelSmoothingLoss(smoothing=0.1, reduction='none'),
        'BCELoss': lambda: nn.BCELoss(reduction='none'),
        'BCEWithLogitsLoss': lambda: nn.BCEWithLogitsLoss(pos_weight=opts.dataset_stats.class_weights, reduction='none')
    }[opts.criterion]()
