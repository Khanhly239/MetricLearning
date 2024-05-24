import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split,  DataLoader

def get_celeba(args):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    print(args["root"])
    
    train_data = datasets.CelebA(
        root=args["root"],
        split='train',
        transform=transform
    )
    
    test_data = datasets.CelebA(
        root=args["root"],
        split='test',
        transform=transform
    )

    val_size = int(len(train_data) * args["val_split"])
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size])
    
    args["num_classes"] = 10
    
    return (train_data, val_data, test_data, args)

args = {
    "root": "/media/mountHDD3/data_storage",
    "val_split": 0.1,
    "h": 64, 
    "num_workers": 4
}

train_ds, valid_ds, test_ds, args = get_celeba(args)
train_ld = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=args["num_workers"])
val_ld = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=args["num_workers"])
test_ld = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=args["num_workers"])
