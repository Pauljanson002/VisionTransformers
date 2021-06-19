import argparse
import datetime
import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import transforms

from dataset import cifar10
from models import ViT


# global data path

# parsing arguments

def get_args_parser():
    parser_inside = argparse.ArgumentParser('Vision transformer training script')

    # Training parameters
    parser_inside.add_argument('--epochs', default=100, type=int)
    return parser_inside


# training loop need to change it to more general one
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, device):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)  # <1>
            labels = labels.to(device=device)
            outputs = model(imgs)
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
    model = ViT().to(device)
    train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    training_loop(args.epochs, optimizer, model, loss_fn, train_loader, device)
    torch.save(model.state_dict(), './state_dicts/vit.pt')
