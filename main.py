import torch
import argparse
from torch import nn
from tqdm import tqdm
from time import time
from os import makedirs
from argparse import Namespace
from os.path import join as join_path
from torch.utils.tensorboard import SummaryWriter

from losses import LabelSmoothingLoss
from setup import get_model, get_optimizer, get_scheduler, get_dataloader, get_epoch_runner_setter, get_criterion
from utils.history import CheckpointState, CheckpointStateOpts, Checkpoint


def load_checkpoint(ckpt_dict, device):
    epoch = ckpt_dict['epoch']
    opts = Namespace(**ckpt_dict['opts'])
    model = get_model(opts).to(device)
    opt = get_optimizer(opts, params=model.parameters())
    model.load_state_dict(ckpt_dict['model_state_dict'])
    opt.load_state_dict(ckpt_dict['optimizer_state_dict'])
    return model, opt, epoch


def load_checkpoint_from_dir(opts):
    with open(join_path(opts.checkpoints_dir, opts.checkpoint), 'rb') as f:
        ckpt_dict = torch.load(f, map_location=device)
        model, opt, last_epoch = load_checkpoint(ckpt_dict, device)
        ckpt = Checkpoint(**ckpt_dict)
    return model, opt, last_epoch, ckpt
    # return model, opt, -1, ckpt


def get_model_state_dict(model):
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    return state_dict


if __name__ == '__main__':
    """
    Good for training datasets 32x32 with 10 classes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--nth_epoch', type=int, default=50)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--patience', type=int, default=1.0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_name', type=str, default='SimpleSASAResNet56')
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--scheduler_name', type=str, default='OneCycleLR')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10')
    parser.add_argument('--outf', type=str, default='checkpoints')
    parser.add_argument('--checkpoint', type=str, default='best.pth')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--n_threads', type=int, default=8, help='Rule of thumb is 4*GPU_AVAIL')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--task', type=str, default='image_classification')
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--criterion', type=str, default='LabelSmoothing')
    opts = parser.parse_args()

    opts.checkpoints_dir = join_path(opts.outf, opts.dataset_name, opts.model_name, opts.scheduler_name)
    makedirs(opts.checkpoints_dir, exist_ok=True)

    # Setup
    timestamp_id = int(time())
    run_name = f'{opts.dataset_name}/{opts.optimizer_name}/{opts.scheduler_name}/{opts.model_name}/{opts.run_id}'
    writer = SummaryWriter(log_dir=f'runs/{run_name}/LR_{opts.lr}_BATCH_{opts.batch_size}/{timestamp_id}')
    device = torch.device(opts.device)
    last_epoch = -1
    ckpt = None
    model = get_model(opts).to(device)
    opt = get_optimizer(opts, params=model.parameters())
    opts.last_epoch = last_epoch

    if opts.resume:
        model, opt, last_epoch, ckpt = load_checkpoint_from_dir(opts)

    scheduler = get_scheduler(opts, opt)
    criterion = get_criterion(opts).to(device)
    start_epoch = max(opts.last_epoch, 0)
    run_epoch = get_epoch_runner_setter(opts)(criterion, device)

    if torch.cuda.device_count() > 0:
        model = nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    # Data
    training_loader = get_dataloader(opts, 'train')
    validation_loader = get_dataloader(opts, 'validation')
    test_loader = get_dataloader(opts, 'test')

    state = CheckpointState(opts=CheckpointStateOpts(
        compare=lambda new, old: new.acc > old.acc,
        patience=opts.patience, max_epochs=opts.epochs,
        ckpt_path=opts.checkpoints_dir, ckpt=ckpt
    ))

    if opts.mode == 'train':
        t = tqdm(range(start_epoch, start_epoch + opts.epochs))
        for i in t:
            if state.is_patience_exhausted():
                break
            train_loss, train_acc = run_epoch(i, model, training_loader, opt, scheduler, is_train=True)
            val_loss, val_acc = run_epoch(i, model, validation_loader, opt, scheduler, is_train=False)
            writer.add_scalars('Loss', {'training': train_loss, 'validation': val_loss}, i)
            writer.add_scalars('Accuracy', {'training': train_acc, 'validation': val_acc}, i)
            ckpt = Checkpoint(
                epoch=i+1, acc=val_acc, loss=val_loss,
                opts=opts.__dict__,
                model=opts.model_name,
                optimizer=opts.optimizer_name,
                model_state_dict=get_model_state_dict(model),
                optimizer_state_dict=opt.state_dict(),
            )
            state.update_best(ckpt)
            state.update_nth(ckpt)
            epoch, acc, loss = state.get_best()
            report = f'Patience(remaining={state.remaining()}) | ' \
                     f'Accuracy(best={acc:.4f}, train={train_acc:.4f}, validation={val_acc:.4f}) | ' \
                     f'Loss(best={loss:.4f}, train={train_loss:.4f}, validation={val_loss:.4f})'
            t.set_description(report, refresh=True)
        writer.close()
    t = tqdm()
    model, _, _, _ = load_checkpoint_from_dir(opts)
    test_loss, test_acc = run_epoch(None, model, test_loader, is_train=False)
    t.set_description(
        f'Test(model={opts.model_name}, optimizer={opts.optimizer_name}, scheduler={opts.scheduler_name}, '
        f'dataset={opts.dataset_name}, loss={test_loss:.4f}, mA={test_acc:.4f}).', refresh=True)
