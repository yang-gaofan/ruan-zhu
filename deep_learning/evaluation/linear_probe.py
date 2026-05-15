import argparse
import csv
import json
import os
from pathlib import Path
import time

import open_clip
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from deep_learning.models.moco_v2 import MocoProjectionHead
from ybjy_config import CACHE_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED_NAME

HF_CACHE_DIR = CACHE_DIR / 'huggingface'
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault('HF_HOME', str(HF_CACHE_DIR))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/final_clip_moco_linear_probe_4060.json')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10'])
    parser.add_argument('--data-root', default='datasets')
    parser.add_argument('--output-dir', default='outputs/final_clip_moco_linear_probe')
    parser.add_argument('--feature-source', default='moco', choices=['clip', 'moco'])
    parser.add_argument('--moco-checkpoint', default='outputs/final_clip_moco_pretrain/moco_v2_last.pt')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--sample-limit', type=int, default=0)
    parser.add_argument('--test-limit', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-download', action='store_true')
    preliminary_args, _ = parser.parse_known_args()
    if preliminary_args.config:
        config_path = Path(preliminary_args.config)
        config_data = json.loads(config_path.read_text(encoding='utf-8'))
        key_map = {
            'data_root': 'data_root',
            'output_dir': 'output_dir',
            'feature_source': 'feature_source',
            'moco_checkpoint': 'moco_checkpoint',
            'batch_size': 'batch_size',
            'learning_rate': 'learning_rate',
            'weight_decay': 'weight_decay',
            'sample_limit': 'sample_limit',
            'test_limit': 'test_limit',
        }
        defaults = {key_map.get(key, key): value for key, value in config_data.items()}
        parser.set_defaults(**defaults)
    return parser.parse_args()


def build_transform(train_mode):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    if train_mode:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def limit_dataset(dataset, sample_limit):
    if sample_limit and sample_limit < len(dataset):
        return Subset(dataset, list(range(sample_limit)))
    return dataset


def build_datasets(args):
    train_dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        transform=build_transform(False),
        download=not args.no_download,
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_root,
        train=False,
        transform=build_transform(False),
        download=not args.no_download,
    )
    return limit_dataset(train_dataset, args.sample_limit), limit_dataset(test_dataset, args.test_limit)


def load_moco_projector(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    projector_state = checkpoint.get('model', {}).get('projector_q')
    if not projector_state:
        raise ValueError(f'MoCo projector not found in {checkpoint_path}')
    first_weight = projector_state['layers.0.weight']
    second_weight = projector_state['layers.2.weight']
    input_dim = int(first_weight.shape[1])
    hidden_dim = int(first_weight.shape[0])
    output_dim = int(second_weight.shape[0])
    projector = MocoProjectionHead(input_dim, hidden_dim, output_dim)
    projector.load_state_dict(projector_state)
    projector.eval()
    for parameter in projector.parameters():
        parameter.requires_grad = False
    return projector, output_dim


def encode_features(clip_model, projector, images, device, amp_enabled):
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            features = clip_model.encode_image(images.to(device, non_blocking=True)).float()
            if projector is not None:
                features = projector(features)
            features = nn.functional.normalize(features, dim=1)
    return features.detach()


def encode_dataset_features(clip_model, projector, loader, device, amp_enabled):
    feature_rows = []
    label_rows = []
    start_time = time.time()
    for images, labels in loader:
        features = encode_features(clip_model, projector, images, device, amp_enabled)
        feature_rows.append(features.cpu())
        label_rows.append(labels.cpu())
    features = torch.cat(feature_rows, dim=0)
    labels = torch.cat(label_rows, dim=0)
    return features, labels, round(time.time() - start_time, 3)


def run_feature_epoch(classifier, loader, criterion, optimizer, device, amp_enabled, train_mode):
    classifier.train(train_mode)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    start_time = time.time()

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                logits = classifier(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    logits = classifier(features)
                    loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += batch_size

    return {
        'loss': round(total_loss / max(1, total_count), 6),
        'accuracy': round(total_correct / max(1, total_count), 6),
        'samples': total_count,
        'seconds': round(time.time() - start_time, 3),
    }


def write_metrics(output_dir, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'linear_probe_metrics.csv'
    json_path = output_dir / 'linear_probe_metrics.json'
    fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy', 'train_seconds', 'test_seconds']
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(metrics, json_file, ensure_ascii=False, indent=2)
    return csv_path, json_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_enabled = args.amp and torch.cuda.is_available()
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    print(f'device={device}', flush=True)
    if torch.cuda.is_available():
        print(f'gpu={torch.cuda.get_device_name(0)}', flush=True)

    train_dataset, test_dataset = build_datasets(args)
    feature_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    feature_test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    print(f'dataset={args.dataset} train={len(train_dataset)} test={len(test_dataset)} source={args.feature_source}', flush=True)

    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED_NAME,
        cache_dir=str(CACHE_DIR / 'open_clip'),
        device=device,
    )
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    projector = None
    feature_dim = 512
    if args.feature_source == 'moco':
        projector, feature_dim = load_moco_projector(args.moco_checkpoint)
        projector = projector.to(device)

    print('encoding train features...', flush=True)
    train_features, train_labels, train_encode_seconds = encode_dataset_features(
        clip_model, projector, feature_train_loader, device, amp_enabled
    )
    print(f'train_features={tuple(train_features.shape)} seconds={train_encode_seconds}', flush=True)
    print('encoding test features...', flush=True)
    test_features, test_labels, test_encode_seconds = encode_dataset_features(
        clip_model, projector, feature_test_loader, device, amp_enabled
    )
    print(f'test_features={tuple(test_features.shape)} seconds={test_encode_seconds}', flush=True)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    classifier = nn.Linear(feature_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metrics = []
    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train_row = run_feature_epoch(classifier, train_loader, criterion, optimizer, device, amp_enabled, True)
        test_row = run_feature_epoch(classifier, test_loader, criterion, optimizer, device, amp_enabled, False)
        scheduler.step()
        best_accuracy = max(best_accuracy, test_row['accuracy'])
        row = {
            'epoch': epoch,
            'train_loss': train_row['loss'],
            'train_accuracy': train_row['accuracy'],
            'test_loss': test_row['loss'],
            'test_accuracy': test_row['accuracy'],
            'train_seconds': train_row['seconds'],
            'test_seconds': test_row['seconds'],
        }
        metrics.append(row)
        print(
            f"epoch={epoch} train_acc={train_row['accuracy']:.4f} test_acc={test_row['accuracy']:.4f} best={best_accuracy:.4f}",
            flush=True,
        )

    csv_path, json_path = write_metrics(output_dir, metrics)
    checkpoint_path = output_dir / 'linear_probe_last.pt'
    torch.save(
        {
            'classifier': classifier.state_dict(),
            'feature_source': args.feature_source,
            'feature_dim': feature_dim,
            'best_accuracy': best_accuracy,
            'metrics': metrics,
            'config': vars(args),
        },
        checkpoint_path,
    )
    print(f'best_accuracy={best_accuracy:.6f}', flush=True)
    print(f'metrics_csv={csv_path}', flush=True)
    print(f'metrics_json={json_path}', flush=True)
    print(f'checkpoint={checkpoint_path}', flush=True)


if __name__ == '__main__':
    main()
