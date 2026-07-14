"""ImageFolder-based dataloaders for the plant disease dataset."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet normalization stats — required since we use ImageNet-pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def get_dataloaders(data_dir: str, batch_size: int = 32, img_size: int = 224, num_workers: int = 2):
    """Expects data_dir/train/<class>/*.jpg and data_dir/valid/<class>/*.jpg."""
    train_tfms, eval_tfms = build_transforms(img_size)

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tfms)
    val_ds = datasets.ImageFolder(f"{data_dir}/valid", transform=eval_tfms)

    # Guard against the two splits somehow disagreeing on class order
    assert train_ds.classes == val_ds.classes, "train/valid class folders don't match"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, train_ds.classes
