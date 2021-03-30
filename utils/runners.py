import torch


def set_image_classification_epoch_runner(criterion, device):
    def run_epoch(epoch, model, loader, optimizer=None, scheduler=None, is_train=False):
        model.eval()
        if is_train:
            model.train()

        iters = len(loader)
        losses, accs = None, None
        for i, (input, target) in enumerate(loader, start=1):
            input, target = input.to(device), target.to(device)
            if is_train:
                for param in model.parameters():
                    param.grad = None
            with torch.set_grad_enabled(is_train):
                y = model(input)
                loss = criterion(y, target)
                acc = (y.argmax(dim=1) == target).half()
                accs = acc.detach() if accs is None else torch.cat((accs, acc.detach()), dim=0)
                losses = loss.detach() if losses is None else torch.cat((losses, loss.detach()), dim=0)
                if is_train:
                    loss.mean().backward()
                    optimizer.step()
                    scheduler.step(epoch + i / iters)

        return loss.mean().item(), accs.mean().item()

    return run_epoch
