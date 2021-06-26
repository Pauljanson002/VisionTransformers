import argparse
import datetime
import math

import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import transforms

from dataset import cifar10
from models import ViT, create_model
from utils import get_optimizer
from models.deit import HardDistillationLoss


# global data path

# parsing arguments

def get_args_parser():
    parser_inside = argparse.ArgumentParser('Vision transformer training script')

    # Training parameters
    parser_inside.add_argument('--epochs', default=100, type=int)
    parser_inside.add_argument('--model', default="vit", type=str)
    parser_inside.add_argument('--savename', default=str(datetime.datetime.now()), type=str)
    parser_inside.add_argument('--optimizer', default='sgd', type=str)
    parser_inside.add_argument('--lr', default=1e-2, type=float)
    parser_inside.add_argument('--warmup', default=5, type=int, metavar='N',
                               help='number of warmup epochs')
    parser_inside.add_argument('--weight_decay', default=0.01, type=float)
    parser_inside.add_argument('--batch_size', default=64, type=int)
    parser_inside.add_argument('--disable-cos', action='store_true',
                               help='disable cosine lr schedule')
    parser_inside.add_argument('--distill', action='store_true')

    return parser_inside


# adjust learning rate
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# training loop need to change it to more general one
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, device,args):
    model.train()
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        adjust_learning_rate(optimizer, epoch, args)
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)  # <1>
            labels = labels.to(device=device)
            outputs = model(imgs)
            if args.distill :
                loss = loss_fn(imgs,outputs,labels)
            else:
                loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))


if __name__ == '__main__':
    # initializing the device and model
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = create_model(args.model).to(device)
    train_loader = torch.utils.data.DataLoader(cifar10, batch_size=args.batch_size, shuffle=True)
    optimizer = get_optimizer(args.optimizer, model.parameters(), learning_rate=args.lr, weight_decay=args.weight_decay)
    if args.distill:
        teacher = create_model('vgg').to(device)
        teacher.load_state_dict(torch.load('./state_dicts/vgg16.pt'))
        loss_fn = HardDistillationLoss(teacher, 0.5)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    training_loop(args.epochs, optimizer, model, loss_fn, train_loader, device,args)
    torch.save(model.state_dict(), './state_dicts/' + args.savename)
