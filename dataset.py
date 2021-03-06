from torchvision import datasets, transforms
from utils.augmentations import CIFAR10Policy

data_path = './data'

augmentations = []
augmentations += [CIFAR10Policy()]
augmentations += [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4915, 0.4823, 0.4468),
        (0.2470, 0.2435, 0.2616)
    )
]
cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose(augmentations)
)
cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4915, 0.4823, 0.4468),
            (0.2470, 0.2435, 0.2616)
        )
    ])
)

cifar100 = datasets.CIFAR100(
    data_path,train=True,download=True,
    transform = transforms.Compose([
        CIFAR10Policy(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])
)

cifar100_val = datasets.CIFAR100(
    data_path,train=False,download=True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])
)