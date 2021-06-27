import argparse
import datetime

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

    parser.add_argument('--savename', type=str, default=str(datetime.datetime.now()))

    # Training parameters

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--model', default='vit', type=str)
    return parser

def accuracy(output:torch.Tensor,labels:torch.Tensor):
    with torch.no_grad():
        batch_size = labels.size(0)
        _,pred = output.topk(1,1,True,True)
        pred = pred.t()
        correct = pred.eq(labels.view(1,-1).expand_as(pred))
        correct_k = correct[:1].flatten().float().sum(0,keepdim=True)
        return correct_k.mul_(100.0 / batch_size)

def train_one_epoch(train_loader, model: torch.nn.Module, loss_fn, optimizer,device, epoch, args):
    model.train()
    total_loss,total_acc = 0,0
    n = 0
    for (images,labels) in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        if args.distill:
            loss = loss_fn(images, output, labels)
        else:
            loss = loss_fn(output, labels)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    model = models.create_model(args.model)
    train_loader = torch.utils.data.DataLoader(dataset.cifar10, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset.cifar10_val, batch_size=args.batch_size, shuffle=False)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    if args.distill:
        teacher = models.create_model('vgg').to(device)
        teacher.load_state_dict(torch.load('./state_dicts/vgg16.pt'))
        loss_fn = models.deit.HardDistillationLoss(teacher, 0.5).to(device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = utils.get_optimizer(args.optimizer, model.parameters(), learning_rate=args.lr, weight_decay=args.weight_decay)
    print("----------------Training starts ---------------------------- ")
    print(f"On device :{device}  Model : {args.model} ")
    start_time = time()
    for epoch in range(1,args.epoch+1):
        train_one_epoch()
