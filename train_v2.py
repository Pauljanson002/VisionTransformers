import argparse
import datetime
import math
import os

import torch

import dataset
import models.deit
import models.model_factory
import utils

from time import time

data_path = '/data'
data_set = 'cifar10'


def get_parser():
    parser = argparse.ArgumentParser(description="Training script")

    # What is workers?

    parser.add_argument('--savename', type=str, default='temp.pt')

    # Training parameters

    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--model', default='vit', type=str)
    return parser


def accuracy(output: torch.Tensor, labels: torch.Tensor):
    with torch.no_grad():
        batch_size = labels.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size)


def train_one_epoch(train_loader, model: torch.nn.Module, loss_fn, optimizer, device, epoch, args):
    model.train()
    total_loss, total_acc = 0, 0
    n = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        if args.distill:
            loss = loss_fn(images, output, labels)
        else:
            loss = loss_fn(output, labels)
        if args.distill:
            acc1 = accuracy(output[0],labels)
        else:
            acc1 = accuracy(output, labels)
        n += images.size(0)
        total_loss += float(loss.item() * images.size(0))
        total_acc += float(acc1 * images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss, avg_acc1 = (total_loss / n), (total_acc / n)
        if i % 10 == 0:
            print(f"Training loss at epoch : {epoch}, Loss = {avg_loss:.4e} , Top-1 {avg_acc1:6.2f}")
    return (total_loss / n), (total_acc / n)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epoch - args.warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate_after_epoch(val_loader, model, loss_fn, device, args, epoch=None, time_begin=None):
    model.eval()
    total_loss, total_accuracy = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)
            acc1 = accuracy(output, labels)
            n += images.size(0)
            total_loss += float(loss.item() * images.size(0))
            total_accuracy += float(acc1 * images.size(0))
            avg_loss, avg_acc1 = (total_loss / n), (total_accuracy / n)
            if i % 10 == 0:
                print(f"Validating loss at epoch : {epoch}, Loss = {avg_loss}, Top-1 {avg_acc1:6.2f}")
    avg_loss, avg_acc1 = (total_loss / n), (total_accuracy / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f"Epoch : {epoch} Top-1 {avg_acc1:6.2f} Time:{total_mins:.2f}")
    return avg_loss, avg_acc1


def plot_data(training_losses, training_accuracies, validation_losses, validation_accuracies, epochs):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1)
    x = [i for i in range(1, epochs + 1)]
    axes[0].plot(x, training_losses, label='training')
    axes[0].plot(x, validation_losses, label='validation')
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('losses')
    axes[0].legend()
    axes[1].plot(x, training_accuracies, label='training')
    axes[1].plot(x, validation_accuracies, label='validation')
    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('accuracies')
    axes[1].legend()
    try:
        fig.savefig('./plots/temporary.png', dpi=fig.dpi)
    except FileNotFoundError:
        fig.savefig('./temporary.png', dpi=fig.dpi)


if __name__ == '__main__':
    best_acc1 = 0
    parser = get_parser()
    args = parser.parse_args()
    model = models.create_model(args.model)
    train_loader = torch.utils.data.DataLoader(dataset.cifar10, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset.cifar10_val, batch_size=args.batch_size, shuffle=False)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)

    # For plotting
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    if args.distill:
        teacher = models.create_model('cct').to(device)
        cwd = os.getcwd()
        teacher.load_state_dict(torch.load('./state_dicts/cct_v4.pt'))
        loss_fn = models.deit.HardDistillationLoss(teacher, 0.5).to(device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = utils.get_optimizer(args.optimizer, model.parameters(), learning_rate=args.learning_rate,
                                    weight_decay=args.weight_decay)

    print("----------------Training starts ---------------------------- ")
    print(f"On device :{device}  Model : {args.model} ")
    start_time = time()
    for epoch in range(1, args.epoch + 1):
        adjust_learning_rate(optimizer, epoch, args)
        avg_training_loss, avg_training_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, device, epoch,
                                                              args)
        if args.distill:
            loss = torch.nn.CrossEntropyLoss().to(device)
            avg_validation_loss, average_validation_acc = validate_after_epoch(test_loader, model, loss, device, args,
                                                                               epoch=epoch, time_begin=start_time)
        else:
            avg_validation_loss, average_validation_acc = validate_after_epoch(test_loader, model, loss_fn, device, args,
                                                                               epoch=epoch, time_begin=start_time)
        best_acc1 = max(average_validation_acc, best_acc1)

        # For plotting
        training_losses.append(avg_training_loss)
        training_accuracies.append(avg_training_acc)
        validation_losses.append(avg_validation_loss)
        validation_accuracies.append(average_validation_acc)

    total_mins = (time() - start_time) / 60
    print(f"Training finished in {total_mins:.2f} mins ")
    print(f"Best top-1 : {best_acc1:.2f} , final top-1:{average_validation_acc:.2f}")
    try:
        torch.save(model.state_dict(), f"./state_dicts/{args.savename}")
    except FileNotFoundError:
        torch.save(model.state_dict(), f'./{args.savename}')
    plot_data(training_losses, training_accuracies, validation_losses, validation_accuracies, epoch)
