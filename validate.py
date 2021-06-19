import torch
import argparse
# todo generalize the dataset
from torchvision import datasets, transforms

from dataset import cifar10, cifar10_val
from models import ViT

def validate(model, train_loader, val_loader):
    accdict = {}
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
    train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64,
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=64,
                                             shuffle=False)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = ViT()
    model.to(device=device)
    model.load_state_dict(torch.load('./state_dicts/vit.pt'))
    print("Model is loaded to %s" % device)
    print("Validation Starting ")
    validate(model,train_loader,val_loader)