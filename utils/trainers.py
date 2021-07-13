import torch
from torch.nn import functional as F
from copy import deepcopy
from os.path import join as join_path
from ignite.engine import Engine, Events
from ignite.metrics import Loss, Fbeta, Accuracy, Precision, Recall

from utils.custom_events import EvaluatorEvents, event_to_attr
from utils.metrics import evaluate_calibration, performance
from utils.log import log_training_loss, log_results, log_calibration_results


def set_image_classification_trainer(model, optimizer, criterion, device, loaders, loggers):
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y).mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)
    loggers['progress_bar'].attach(trainer, metric_names='all')

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, target = batch[0].to(device), batch[1].to(device)
            y = model(x)
            return {'y_pred': y, 'y': target, 'criterion_kwargs': {}}

    evaluator = Engine(validation_step)
    evaluator.state.validation_completed = 0
    evaluator.register_events(*EvaluatorEvents, event_to_attr=event_to_attr)

    metrics = {
        'loss': Loss(criterion),
        'F1': Fbeta(beta=1, average=False),
        'mA': Accuracy(is_multilabel=False),
        'mP': Precision(average=False, is_multilabel=False),
        'mR': Recall(average=False, is_multilabel=False)
    }
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=250), log_training_loss, loggers)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        with evaluator.add_event_handler(Events.COMPLETED, log_results, 'train', engine.state.epoch, loggers):
            evaluator.run(loaders['train'])
        with evaluator.add_event_handler(Events.COMPLETED, log_results, 'validation', engine.state.epoch, loggers):
            evaluator.run(loaders['validation'])
            evaluator.state.validation_completed += 1
            evaluator.fire_event(EvaluatorEvents.VALIDATION_COMPLETED)

    @trainer.on(Events.COMPLETED)
    def test(engine):
        with evaluator.add_event_handler(
            Events.COMPLETED, log_results, 'test', engine.state.epoch, loggers
        ), evaluator.add_event_handler(
            Events.COMPLETED, log_calibration_results, 'test', loggers,
            output_transform=lambda output: {'y_pred': F.softmax(output['y_pred'], dim=1), 'y': output['y']}
        ):
            evaluator.run(loaders['test'])

    return trainer, evaluator


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
