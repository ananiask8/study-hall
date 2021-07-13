from utils.metrics import evaluate_calibration


def log_training_loss(trainer, loggers):
    m = f'Training(epoch={trainer.state.epoch}, step={trainer.state.iteration}' \
        f'/{trainer.state.epoch_length}, loss={trainer.state.output:.4f})'
    loggers['progress_bar'].log_message(m)


def log_results(engine, mode, epoch, loggers):
    metrics = engine.state.metrics
    Loss = metrics['loss']
    F1, mA, mP, mR = metrics['F1'].mean(), metrics['mA'], metrics['mP'].mean(), metrics['mR'].mean()
    m = f'Evaluation(mode={mode}, F1={F1:.4f}, mA={mA:.4f}, mP={mP:.4f}, mR={mR:.4f})'
    loggers['progress_bar'].log_message(m)
    loggers['tensorboard'].add_scalars(f'metrics/{mode}', {
        'Loss': Loss, 'F1': F1, 'mA': mA, 'mP': mP, 'mR': mR
    }, epoch)


def log_calibration_results(engine, mode, loggers, output_transform=None):
    output = engine.state.output
    if output_transform is not None:
        output = output_transform(engine.state.output)
    preds = output['y_pred']
    targets = output['y']
    accs = (preds.argmax(dim=1) == targets).half()
    fig = evaluate_calibration(preds.max(dim=1)[0].cpu().numpy(), accs.cpu().numpy(), 'model')
    loggers['tensorboard'].add_figure('reliability_curve', fig)
