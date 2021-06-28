import torch
from os.path import join as join_path

from utils.metrics import evaluate_calibration, performance


def set_image_classification_epoch_runner(criterion, device, opts):
    def run_epoch(epoch, model, loader, optimizer=None, scheduler=None, mode='train'):
        is_train = mode == 'train'
        is_test = mode == 'test'
        model.eval()
        if is_train:
            model.train()

        iters = len(loader)
        preds, targets, losses, accs = None, None, None, None
        for i, (input, target) in enumerate(loader, start=1):
            input, target = input.to(device), target.to(device)
            if is_train:
                for param in model.parameters():
                    param.grad = None
            with torch.set_grad_enabled(is_train):
                pred = model(input)
                loss = criterion(pred, target)
                acc = (pred.argmax(dim=1) == target).half()
                pred = torch.softmax(pred, dim=1)
                pred = pred.max(dim=1)[0]
                preds = pred.detach() if preds is None else torch.cat((preds, pred.detach()), dim=0)
                targets = target.detach() if targets is None else torch.cat((targets, target.detach()), dim=0)
                accs = acc.detach() if accs is None else torch.cat((accs, acc.detach()), dim=0)
                losses = loss.detach() if losses is None else torch.cat((losses, loss.detach()), dim=0)
                if is_train:
                    loss.mean().backward()
                    optimizer.step()
                    scheduler.step(epoch + i / iters)
        if is_test:
            filename = join_path(opts.checkpoints_dir, 'reliability_curve.png')
            # we take p(y = target) = p(yi) (this is the reason why we pass the accuracy down below)
            #         / p(yi) -> 1 if y == target
            # we want—
            #         \ p(yi) -> 0 if y != target
            evaluate_calibration(preds.cpu().numpy(), accs.cpu().numpy(), opts.model_name, filename)
        return losses.mean().item(), accs.mean().item(), -1.

    return run_epoch


def set_adult_census_epoch_runner(criterion, device, opts):
    def run_epoch(epoch, model, loader, optimizer=None, scheduler=None, mode='train'):
        is_train = mode == 'train'
        is_test = mode == 'test'
        model.eval()
        if is_train:
            model.train()

        iters = len(loader)
        preds, targets, losses, accs = None, None, None, None
        for i, (input, target) in enumerate(loader, start=1):
            input, target = input.to(device), target.to(device)
            if is_train:
                for param in model.parameters():
                    param.grad = None
            with torch.set_grad_enabled(is_train):
                pred, features = model(input)
                pred = pred.view(-1)
                target = target.view(-1)
                loss = criterion(pred, target)
                pred = torch.sigmoid(pred)
                y = pred.round()
                acc = (y == target).half()
                preds = pred.detach() if preds is None else torch.cat((preds, pred.detach()), dim=0)
                targets = target.detach() if targets is None else torch.cat((targets, target.detach()), dim=0)
                accs = acc.detach() if accs is None else torch.cat((accs, acc.detach()), dim=0)
                losses = loss.detach() if losses is None else torch.cat((losses, loss.detach()), dim=0)
                if is_train:
                    loss.mean().backward()
                    optimizer.step()
                    scheduler.step(epoch + i / iters)
        m = performance(preds.cpu().numpy(), targets.cpu().numpy())
        if is_test:
            filename = join_path(opts.checkpoints_dir, 'reliability_curve.png')
            evaluate_calibration(preds.cpu().numpy(), targets.cpu().numpy(), opts.model_name, filename)
        return losses.mean().item(), accs.mean().item(), m.f1score

    return run_epoch
