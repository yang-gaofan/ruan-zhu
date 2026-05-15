import argparse
import csv
import json
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

import open_clip

from deep_learning.data.datasets import build_clip_moco_dataset, build_moco_dataset
from deep_learning.models.moco_v2 import FrozenClipMocoV2, MocoV2, save_moco_checkpoint
from ybjy_config import CACHE_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED_NAME

HF_CACHE_DIR = CACHE_DIR / 'huggingface'
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault('HF_HOME', str(HF_CACHE_DIR))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'stl10', 'image_folder'])
    parser.add_argument('--data-root', default='datasets')
    parser.add_argument('--image-dir', default='')
    parser.add_argument('--output-dir', default='outputs/moco_v2')
    parser.add_argument('--encoder', default='frozen_clip', choices=['frozen_clip', 'resnet18'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--queue-size', type=int, default=1024)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--sample-limit', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-download', action='store_true')
    preliminary_args, _ = parser.parse_known_args()
    if preliminary_args.config:
        config_path = Path(preliminary_args.config)
        config_data = json.loads(config_path.read_text(encoding='utf-8'))
        key_map = {
            'learning_rate': 'lr',
            'data_root': 'data_root',
            'image_dir': 'image_dir',
            'output_dir': 'output_dir',
            'batch_size': 'batch_size',
            'queue_size': 'queue_size',
            'feature_dim': 'feature_dim',
            'sample_limit': 'sample_limit',
            'log_every': 'log_every',
        }
        defaults = {}
        for key, value in config_data.items():
            target_key = key_map.get(key, key)
            defaults[target_key] = value
        parser.set_defaults(**defaults)
    return parser.parse_args()


def build_optimizer(model, lr):
    parameters = model.trainable_parameters() if hasattr(model, 'trainable_parameters') else model.encoder_q.parameters()
    return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.0001)


def build_training_model(args, device):
    if args.encoder == 'frozen_clip':
        clip_model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED_NAME,
            cache_dir=str(CACHE_DIR / 'open_clip'),
            device=device,
        )
        clip_model.eval()
        return FrozenClipMocoV2(
            clip_model=clip_model,
            feature_dim=args.feature_dim,
            queue_size=args.queue_size,
            momentum=args.momentum,
            temperature=args.temperature,
        ).to(device)
    return MocoV2(
        feature_dim=args.feature_dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
    ).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, max_batches, log_every):
    model.train()
    total_loss = 0.0
    total_count = 0
    start_time = time.time()
    amp_enabled = torch.cuda.is_available() and getattr(model, 'use_amp', False)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    for batch_index, ((image_q, image_k), _) in enumerate(loader, start=1):
        image_q = image_q.to(device, non_blocking=True)
        image_k = image_k.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            logits, labels = model(image_q, image_k)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = image_q.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        current_loss = total_loss / max(1, total_count)
        if batch_index == 1 or batch_index % log_every == 0:
            print(
                f'epoch={epoch} batch={batch_index} samples={total_count} loss={current_loss:.6f}',
                flush=True,
            )
        if max_batches and batch_index >= max_batches:
            break

    elapsed = time.time() - start_time
    return {
        'epoch': epoch,
        'loss': round(total_loss / max(1, total_count), 6),
        'samples': total_count,
        'seconds': round(elapsed, 3),
    }


def write_metrics(output_dir, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'metrics.csv'
    json_path = output_dir / 'metrics.json'
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['epoch', 'loss', 'samples', 'seconds'])
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(metrics, json_file, ensure_ascii=False, indent=2)
    return csv_path, json_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    if torch.cuda.is_available():
        print(f'gpu={torch.cuda.get_device_name(0)}', flush=True)

    sample_limit = args.sample_limit
    if args.encoder == 'frozen_clip':
        dataset = build_clip_moco_dataset(
            name=args.dataset,
            root=args.data_root,
            download=not args.no_download,
            sample_limit=sample_limit,
            image_dir=args.image_dir,
        )
    else:
        dataset = build_moco_dataset(
            name=args.dataset,
            root=args.data_root,
            download=not args.no_download,
            sample_limit=sample_limit,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    print(f'dataset={args.dataset} encoder={args.encoder} size={len(dataset)} batches={len(loader)}', flush=True)

    model = build_training_model(args, device)
    model.use_amp = args.amp
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(model, args.lr)

    metrics = []
    checkpoint_path = output_dir / 'moco_v2_last.pt'
    for epoch in range(1, args.epochs + 1):
        metrics.append(train_one_epoch(model, loader, criterion, optimizer, device, epoch, args.max_batches, args.log_every))
        write_metrics(output_dir, metrics)
        save_moco_checkpoint(checkpoint_path, model, optimizer, epoch, metrics)

    csv_path, json_path = write_metrics(output_dir, metrics)
    save_moco_checkpoint(checkpoint_path, model, optimizer, args.epochs, metrics)
    print(f'metrics_csv={csv_path}', flush=True)
    print(f'metrics_json={json_path}', flush=True)
    print(f'checkpoint={checkpoint_path}', flush=True)


if __name__ == '__main__':
    main()
