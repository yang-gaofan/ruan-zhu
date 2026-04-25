import argparse
import csv
import json
import os
from pathlib import Path

import open_clip
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from PIL import Image

from ybjy_config import CACHE_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED_NAME

HF_CACHE_DIR = CACHE_DIR / 'huggingface'
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault('HF_HOME', str(HF_CACHE_DIR))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'image_folder'])
    parser.add_argument('--data-root', default='datasets')
    parser.add_argument('--image-dir', default='')
    parser.add_argument('--output-dir', default='outputs/zero_shot_cifar10')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sample-limit', type=int, default=0)
    parser.add_argument('--export-top-per-class', type=int, default=3)
    preliminary_args, _ = parser.parse_known_args()
    if preliminary_args.config:
        config_data = json.loads(Path(preliminary_args.config).read_text(encoding='utf-8'))
        key_map = {
            'data_root': 'data_root',
            'output_dir': 'output_dir',
            'batch_size': 'batch_size',
            'sample_limit': 'sample_limit',
            'export_top_per_class': 'export_top_per_class',
        }
        defaults = {}
        for key, value in config_data.items():
            defaults[key_map.get(key, key)] = value
        parser.set_defaults(**defaults)
    return parser.parse_args()


def build_dataset(args, preprocess):
    if args.dataset == 'image_folder':
        image_dir = Path(args.image_dir)
        image_paths = sorted(
            [
                path
                for path in image_dir.iterdir()
                if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            ]
        )
        return ImageFolderZeroShotDataset(image_paths, preprocess), None
    dataset = datasets.CIFAR10(root=args.data_root, train=False, transform=preprocess, download=True)
    raw_dataset = datasets.CIFAR10(root=args.data_root, train=False, transform=None, download=False)
    if args.sample_limit and args.sample_limit < len(dataset):
        indices = list(range(args.sample_limit))
        dataset = Subset(dataset, indices)
        raw_dataset = Subset(raw_dataset, indices)
    return dataset, raw_dataset


class ImageFolderZeroShotDataset:
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.classes = sorted({self.class_name_from_path(path) for path in image_paths})
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert('RGB')
        label_name = self.class_name_from_path(path)
        return self.preprocess(image), self.class_to_index[label_name]

    def class_name_from_path(self, path):
        raw_name = path.stem.split('_', 1)[1] if '_' in path.stem else path.stem
        return raw_name.replace('_', ' ').replace('-', ' ')


def build_text_features(model, tokenizer, class_names, device):
    templates = [
        'a photo of a {}.',
        'a clear photo of a {}.',
        'a small photo of a {}.',
        'a cropped photo of a {}.',
    ]
    class_features = []
    with torch.no_grad():
        for class_name in class_names:
            texts = [template.format(class_name.replace('_', ' ')) for template in templates]
            tokens = tokenizer(texts).to(device)
            text_feature = model.encode_text(tokens)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature.mean(dim=0)
            text_feature = text_feature / text_feature.norm()
            class_features.append(text_feature)
    return torch.stack(class_features, dim=1)


def export_ranked_images(raw_dataset, rows, class_names, output_dir, top_per_class):
    export_dir = output_dir / 'selected_demo_images'
    export_dir.mkdir(parents=True, exist_ok=True)
    per_class = {index: 0 for index in range(len(class_names))}
    for row in sorted(rows, key=lambda item: item['confidence'], reverse=True):
        label = row['label']
        if row['prediction'] != label:
            continue
        if per_class[label] >= top_per_class:
            continue
        image, _ = raw_dataset[row['index']]
        file_name = f"{label:02d}_{per_class[label] + 1:02d}_{class_names[label]}_{row['confidence']:.3f}.png"
        image.resize((384, 384)).save(export_dir / file_name)
        per_class[label] += 1
        if all(value >= top_per_class for value in per_class.values()):
            break
    return export_dir


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    if torch.cuda.is_available():
        print(f'gpu={torch.cuda.get_device_name(0)}', flush=True)

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED_NAME,
        cache_dir=str(CACHE_DIR / 'open_clip'),
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    dataset, raw_dataset = build_dataset(args, preprocess)
    class_names = dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes
    text_features = build_text_features(model, tokenizer, class_names, device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )

    rows = []
    correct1 = 0
    correct5 = 0
    total = 0
    offset = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features
            probabilities = logits.softmax(dim=-1)
            top_values, top_indices = probabilities.topk(5, dim=-1)
            correct1 += int((top_indices[:, 0] == labels).sum().item())
            correct5 += int((top_indices == labels.unsqueeze(1)).any(dim=1).sum().item())
            batch_size = images.shape[0]
            for local_index in range(batch_size):
                rows.append(
                    {
                        'index': offset + local_index,
                        'label': int(labels[local_index].item()),
                        'label_name': class_names[int(labels[local_index].item())],
                        'prediction': int(top_indices[local_index, 0].item()),
                        'prediction_name': class_names[int(top_indices[local_index, 0].item())],
                        'confidence': round(float(top_values[local_index, 0].item()), 6),
                    }
                )
            total += batch_size
            offset += batch_size

    metrics = {
        'dataset': args.dataset,
        'samples': total,
        'top1': round(correct1 / max(1, total), 6),
        'top5': round(correct5 / max(1, total), 6),
        'model': CLIP_MODEL_NAME,
        'pretrained': CLIP_PRETRAINED_NAME,
    }
    with open(output_dir / 'zero_shot_metrics.json', 'w', encoding='utf-8') as json_file:
        json.dump(metrics, json_file, ensure_ascii=False, indent=2)
    with open(output_dir / 'zero_shot_predictions.csv', 'w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    export_dir = output_dir
    if raw_dataset is not None:
        export_dir = export_ranked_images(raw_dataset, rows, class_names, output_dir, args.export_top_per_class)
    print(f"top1={metrics['top1']}", flush=True)
    print(f"top5={metrics['top5']}", flush=True)
    print(f'metrics={output_dir / "zero_shot_metrics.json"}', flush=True)
    print(f'predictions={output_dir / "zero_shot_predictions.csv"}', flush=True)
    print(f'images={export_dir}', flush=True)


if __name__ == '__main__':
    main()
