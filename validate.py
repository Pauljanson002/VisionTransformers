

import torch
import argparse
# todo generalize the dataset
from torchvision import datasets, transforms

from dataset import cifar10, cifar10_val
from models import ViT, create_model


def get_arg_parser():
    parser = argparse.ArgumentParser('Vision transformers validation parser')

    # what to validate
    parser.add_argument('--model', default='vit', type=str)
    parser.add_argument('--path', default='vit.pt', type=str)
    return parser


# inspired by Shi labs
def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return  res

def validate_ver2(model, train_loader, device=torch.device('cpu')):
    model.eval()
    acc1_val = 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device=device)
            target = target.to(device=device)
            output = model(images)
            acc1 = accuracy(output, target)
            n += images.size(0)
            acc1_val += float(acc1[0] * images.size(0))
        avg_acc1 = (acc1_val / n)

        return avg_acc1


def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)  # <1>
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name, correct / total))


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    train_loader = torch.utils.data.DataLoader(cifar10, batch_size=128,
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=128,
                                             shuffle=False)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = create_model(args.model)
    model.to(device=device)
    model.load_state_dict(torch.load('./state_dicts/' + args.path))
    print("Model is loaded to %s" % device)
    print("Validation Starting ")
    validate(model, train_loader, val_loader)
    model.eval()
    acc1 = validate_ver2(model,train_loader,device=device)
    acc2 = validate_ver2(model,val_loader,device=device)
    print(acc1)
    print(f"Test accuracy {acc2:.2f}")
