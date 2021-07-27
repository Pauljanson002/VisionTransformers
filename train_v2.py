import argparse
import datetime
import math
import os

import torch
import wandb
from torchvision import datasets, transforms
from utils.augmentations import CIFAR10Policy
import dataset
import models.deit
import models.model_factory
import utils
from tqdm import tqdm
from time import time

torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)

data_path = '/data'


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
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
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
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        if args.distill:
            loss = loss_fn(images, output, labels)
        else:
            loss = loss_fn(output, labels)
        if args.distill:
            acc1 = accuracy(output[0], labels)
        else:
            acc1 = accuracy(output, labels)
        n += images.size(0)
        total_loss += float(loss.item() * images.size(0))
        total_acc += float(acc1 * images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # avg_loss, avg_acc1 = (total_loss / n), (total_acc / n)
        # if i % 10 == 0:
        #     print(f"Training loss at epoch : {epoch}, Loss = {avg_loss:.4e} , Top-1 {avg_acc1:6.2f}")
    return (total_loss / n), (total_acc / n)


def adjust_learning_rate(optimizer, epoch, learning_rate, final_epoch, warmup=0):
    lr = learning_rate
    if warmup > 0 and epoch < warmup:
        lr = lr / (warmup - epoch)
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup) / (final_epoch - warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate_after_epoch(val_loader, model, loss_fn, device, args, epoch=None, time_begin=None):
    model.eval()
    total_loss, total_accuracy = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)
            acc1 = accuracy(output, labels)
            n += images.size(0)
            total_loss += float(loss.item() * images.size(0))
            total_accuracy += float(acc1 * images.size(0))
            # avg_loss, avg_acc1 = (total_loss / n), (total_accuracy / n)
            # if i % 10 == 0:
            #     print(f"Validating loss at epoch : {epoch}, Loss = {avg_loss}, Top-1 {avg_acc1:6.2f}")
    avg_loss, avg_acc1 = (total_loss / n), (total_accuracy / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f"Epoch : {epoch} Top-1 {avg_acc1:6.2f} Time:{total_mins:.2f}")
    return avg_loss, avg_acc1


if __name__ == '__main__':
    best_acc1 = 0
    parser = get_parser()
    args = parser.parse_args()
    wandb.init(
        project="Cifar100 on VitLite",
        entity="Pauljanson002"
    )
    config = wandb.config
    config.args = args
    model = models.create_model(args.model)
    if args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(dataset.cifar100, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(dataset.cifar100_val, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers)
    else:
        train_loader = torch.utils.data.DataLoader(dataset.cifar10, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(dataset.cifar10_val, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers)
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
    initial_epoch = 1
    if args.resume != '':
        checkpoint = torch.load(f"./checkpoints/{args.resume}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch']
        last_acc1 = checkpoint['accuracy']

    print("----------------Training starts ---------------------------- ")
    print(f"On device :{device}  Model : {args.model} ")
    start_time = time()
    if args.epoch < initial_epoch:
        initial_epoch = 1
    for epoch in range(initial_epoch, args.epoch + 1):
        adjust_learning_rate(optimizer, epoch, args.learning_rate, args.epoch, args.warmup)
        avg_training_loss, avg_training_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, device, epoch,
                                                              args)
        if args.distill:
            loss = torch.nn.CrossEntropyLoss().to(device)
            avg_validation_loss, average_validation_acc = validate_after_epoch(test_loader, model, loss, device, args,
                                                                               epoch=epoch, time_begin=start_time)
        else:
            avg_validation_loss, average_validation_acc = validate_after_epoch(test_loader, model, loss_fn, device,
                                                                               args,
                                                                               epoch=epoch, time_begin=start_time)
        best_acc1 = max(average_validation_acc, best_acc1)
        wandb.log({
            'training_accuracy': avg_training_acc,
            'training_loss': avg_training_loss,
            'validation_accuracy': average_validation_acc,
            'validation_loss': avg_validation_loss
        })

    total_mins = (time() - start_time) / 60
    print(f"Training finished in {total_mins:.2f} mins ")
    print(f"Best top-1 : {best_acc1:.2f} , final top-1:{average_validation_acc:.2f}")
    model_dict = {
        'epoch': args.epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': best_acc1
    }
    try:
        torch.save(model_dict, f"./checkpoints/{args.savename}")
    except FileNotFoundError:
        torch.save(model_dict, f'./{args.savename}')
