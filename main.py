import copy
import time

import torch
import torch.nn as nn

import dataset
from models import create_model
from utils import get_optimizer
from torch.optim import lr_scheduler

accuracies = []

def train_model(model, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loader = torch.utils.data.DataLoader(dataset.cifar10, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset.cifar10_val, batch_size=128, shuffle=False)
        dataloaders = {
            'train': train_loader,
            'val': test_loader}
        n_train, n_val = 0, 0
        dataset_sizes = {}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    n_train += inputs.size(0)
                    dataset_sizes[phase] = n_train
                else:
                    n_val += inputs.size(0)
                    dataset_sizes[phase] = n_val
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    accuracies.append(best_acc.item())
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def fine_tune(limit=14):
    model = create_model('vit_lite_h')
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    checkpoint = torch.load(f"./checkpoints/vit_lite_h_200.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model : is loaded to {device}")

    for param in model.parameters():
        param.requires_grad = False
    fc_in_features = model.classifier.fc.in_features
    fc_out_features = model.classifier.fc.out_features
    model.classifier.fc = nn.Linear(fc_in_features, fc_out_features).to(device)
    model.classifier.blocks = model.classifier.blocks[:limit]
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer('adamw', model.parameters(), 0.001, 3e-2)
    cos_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 100)
    print(model)
    model_fine_tuned = train_model(model, criterion, optimizer, cos_lr_scheduler, device, 20)


if __name__ == '__main__':
    print("Fine tuning script")
    for i in range(32,0,-1):
        print(f"Number of transformer layers Active {i}")
        fine_tune(limit=i)
    print(accuracies)