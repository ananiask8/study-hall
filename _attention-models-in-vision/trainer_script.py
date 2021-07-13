import torch
import argparse
from torch import nn
from time import time
from os import makedirs
from os.path import join as join_path
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.contrib.handlers import ProgressBar, FastaiLRFinder
from ignite.engine import Events, Engine

from trainer_setup import get_tensorboard_writer, get_model, get_optimizer, get_scheduler,\
    get_dataloader, get_criterion, get_score_function, get_trainer_evaluator
from utils.custom_events import EvaluatorEvents
from utils.stats import get_dataset_stats


if __name__ == '__main__':
    """
    Good for training datasets 32x32 with 10 classes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=int, default=1e-5)
    parser.add_argument('--patience_factor', type=int, default=2.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='Baseline')
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--scheduler_name', type=str, default='OneCycleLR')
    parser.add_argument('--dataset_name', type=str, default='small.CIFAR10')
    parser.add_argument('--outf', type=str, default='checkpoints')
    parser.add_argument('--n_threads', type=int, default=4, help='Rule of thumb is 4*GPU_AVAIL')
    parser.add_argument('--task', type=str, default='image_classification')
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--criterion', type=str, default='LabelSmoothingCrossEntropyLoss')
    parser.add_argument('--score_name', type=str, default='F1')
    opts = parser.parse_args()

    opts.timestamp = int(time())
    opts.checkpoints_dir = join_path(opts.outf, str(opts.timestamp))
    makedirs(opts.checkpoints_dir, exist_ok=True)

    # Setup
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
    writer = get_tensorboard_writer(opts)
    model = get_model(opts).to(device)
    optimizer = get_optimizer(opts, params=model.parameters())
    score_function = get_score_function(opts)
    loggers = {
        'tensorboard': writer,
        'progress_bar': ProgressBar(persist=True)
    }

    # Data
    loaders = {
        'train': get_dataloader(opts, 'train'),
        'validation': get_dataloader(opts, 'validation'),
        'test': get_dataloader(opts, 'test')
    }
    opts.dataset_stats = get_dataset_stats(loaders['train'].dataset)
    criterion = get_criterion(opts).to(device)

    # TODO Replace with Ignite parallelization
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    trainer, evaluator = get_trainer_evaluator(opts)(model, optimizer, criterion, device, loaders, loggers)
    raw_trainer = Engine(trainer._process_function)
    # Handlers
    lr_finder = FastaiLRFinder()
    to_save = {'model': model, 'optimizer': optimizer}
    with lr_finder.attach(raw_trainer, to_save=to_save) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(loaders['train'])
        lr_finder.get_results()
        lr_finder.plot()
        opts.lr = lr_finder.lr_suggestion()

    opts.total_steps = len(loaders['train']) * opts.epochs
    scheduler = get_scheduler(opts, optimizer)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

    to_save['scheduler'] = scheduler
    save_handler = Checkpoint(
        to_save, DiskSaver(opts.checkpoints_dir),
        n_saved=2, filename_prefix='best',
        score_function=score_function, score_name=opts.score_name
    )
    evaluator.add_event_handler(EvaluatorEvents.VALIDATION_COMPLETED, save_handler)

    patience = opts.patience_factor * opts.epochs
    early_stopping_handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(EvaluatorEvents.VALIDATION_COMPLETED, early_stopping_handler)

    # Run
    trainer.run(loaders['train'], max_epochs=opts.epochs)
