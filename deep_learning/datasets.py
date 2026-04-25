from torch.utils.data import Subset
from torchvision import datasets, transforms


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image):
        return self.base_transform(image), self.base_transform(image)


def build_cifar10_moco_transform(image_size=32):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )


def build_stl10_moco_transform(image_size=96):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ]
    )


def build_clip_moco_transform(image_size=224):
    color_jitter = transforms.ColorJitter(0.32, 0.32, 0.32, 0.08)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.35, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.7),
            transforms.RandomGrayscale(p=0.12),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def build_moco_dataset(name, root, download=True, sample_limit=0):
    normalized_name = name.lower()
    if normalized_name == 'cifar10':
        transform = TwoCropsTransform(build_cifar10_moco_transform())
        dataset = datasets.CIFAR10(root=root, train=True, transform=transform, download=download)
    elif normalized_name == 'stl10':
        transform = TwoCropsTransform(build_stl10_moco_transform())
        dataset = datasets.STL10(root=root, split='unlabeled', transform=transform, download=download)
    elif normalized_name == 'image_folder':
        transform = TwoCropsTransform(build_stl10_moco_transform())
        dataset = datasets.ImageFolder(root=root, transform=transform)
    else:
        raise ValueError(f'Unsupported dataset: {name}')
    if normalized_name in ['cifar10', 'stl10'] and sample_limit and sample_limit < len(dataset):
        dataset.data = dataset.data[:sample_limit]
        if hasattr(dataset, 'labels') and dataset.labels is not None:
            dataset.labels = dataset.labels[:sample_limit]
        return dataset
    return _limit_dataset(dataset, sample_limit)


def _limit_dataset(dataset, sample_limit):
    if sample_limit and sample_limit < len(dataset):
        return Subset(dataset, list(range(sample_limit)))
    return dataset


def build_clip_moco_dataset(name, root, download=True, sample_limit=0, image_dir=''):
    normalized_name = name.lower()
    transform = TwoCropsTransform(build_clip_moco_transform())
    if normalized_name == 'cifar10':
        dataset = datasets.CIFAR10(root=root, train=True, transform=transform, download=download)
    elif normalized_name == 'stl10':
        dataset = datasets.STL10(root=root, split='unlabeled', transform=transform, download=download)
    elif normalized_name == 'image_folder':
        dataset = datasets.ImageFolder(root=image_dir or root, transform=transform)
    else:
        raise ValueError(f'Unsupported dataset: {name}')
    return _limit_dataset(dataset, sample_limit)
