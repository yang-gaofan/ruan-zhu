import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from ybjy_config import EXPORT_DIR
from app.ybjy_database import (
    list_ybjy_feature_experiments_by_project,
    list_ybjy_samples_by_project,
    save_ybjy_feature_experiment,
)


def _load_ybjy_feature_matrix(project_id):
    sample_rows = list_ybjy_samples_by_project(project_id)
    feature_rows = []
    for sample in sample_rows:
        try:
            feature_value = json.loads(sample.get('feature_json', '[]'))
        except json.JSONDecodeError:
            continue
        if not isinstance(feature_value, list) or not feature_value:
            continue
        feature_rows.append(
            {
                'id': sample['id'],
                'file_name': sample['file_name'],
                'relative_path': sample['relative_path'],
                'feature': feature_value,
            }
        )
    if not feature_rows:
        return feature_rows, None
    feature_array = np.array([row['feature'] for row in feature_rows], dtype=np.float32)
    return feature_rows, feature_array


def _normalize_numpy_matrix(feature_array):
    norm_value = np.linalg.norm(feature_array, axis=1, keepdims=True)
    norm_value[norm_value == 0] = 1.0
    return feature_array / norm_value


def build_ybjy_feature_queue_summary(project_id, queue_size):
    feature_rows, feature_array = _load_ybjy_feature_matrix(project_id)
    if feature_array is None:
        return {
            'sample_count': 0,
            'feature_dim': 0,
            'queue_size': queue_size,
            'used_queue_size': 0,
            'mean_similarity': 0.0,
            'max_similarity': 0.0,
            'min_similarity': 0.0,
            'candidate_rows': [],
        }

    normalized = _normalize_numpy_matrix(feature_array)
    used_queue_size = min(queue_size, len(normalized))
    queue_matrix = normalized[:used_queue_size]
    similarity = normalized @ queue_matrix.T

    candidate_rows = []
    for row_index, row in enumerate(feature_rows[:min(12, len(feature_rows))]):
        current_scores = similarity[row_index].copy()
        if row_index < used_queue_size:
            current_scores[row_index] = -1.0
        best_index = int(np.argmax(current_scores))
        best_score = float(current_scores[best_index])
        candidate_rows.append(
            {
                'sample_id': row['id'],
                'file_name': row['file_name'],
                'nearest_name': feature_rows[best_index]['file_name'],
                'nearest_score': round(best_score, 6),
            }
        )

    flat_scores = similarity.reshape(-1)
    return {
        'sample_count': len(feature_rows),
        'feature_dim': int(feature_array.shape[1]),
        'queue_size': queue_size,
        'used_queue_size': used_queue_size,
        'mean_similarity': round(float(np.mean(flat_scores)), 6),
        'max_similarity': round(float(np.max(flat_scores)), 6),
        'min_similarity': round(float(np.min(flat_scores)), 6),
        'candidate_rows': candidate_rows,
    }


def _make_feature_views(feature_tensor, noise_scale):
    first_noise = torch.randn_like(feature_tensor) * noise_scale
    second_noise = torch.randn_like(feature_tensor) * noise_scale
    first_view = functional.normalize(feature_tensor + first_noise, dim=1)
    second_view = functional.normalize(feature_tensor + second_noise, dim=1)
    return first_view, second_view


def _build_contrastive_loss(first_projection, second_projection, temperature_value):
    batch_size = first_projection.shape[0]
    logits = first_projection @ second_projection.T
    logits = logits / temperature_value
    labels = torch.arange(batch_size, device=first_projection.device)
    first_loss = functional.cross_entropy(logits, labels)
    second_loss = functional.cross_entropy(logits.T, labels)
    return (first_loss + second_loss) / 2.0


def run_ybjy_projection_head_training(project_id, train_options):
    feature_rows, feature_array = _load_ybjy_feature_matrix(project_id)
    if feature_array is None or len(feature_rows) < 2:
        return {
            'ok': False,
            'message': '当前项目至少需要 2 张已提取特征的图片。',
        }

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    torch.manual_seed(2026)
    np.random.seed(2026)

    normalized = _normalize_numpy_matrix(feature_array)
    feature_tensor = torch.tensor(normalized, dtype=torch.float32, device=device)
    input_dim = feature_tensor.shape[1]
    projection_dim = int(train_options['projection_dim'])
    epoch_count = int(train_options['epoch_count'])
    learning_rate = float(train_options['learning_rate'])
    temperature_value = float(train_options['temperature_value'])
    noise_scale = float(train_options['noise_scale'])

    projection = torch.nn.Sequential(
        torch.nn.Linear(input_dim, projection_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(projection_dim, projection_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=learning_rate, weight_decay=0.0001)

    metric_rows = []
    final_loss = 0.0
    for epoch_index in range(1, epoch_count + 1):
        first_view, second_view = _make_feature_views(feature_tensor, noise_scale)
        first_projection = functional.normalize(projection(first_view), dim=1)
        second_projection = functional.normalize(projection(second_view), dim=1)
        loss_value = _build_contrastive_loss(first_projection, second_projection, temperature_value)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        final_loss = float(loss_value.detach().cpu().item())
        metric_rows.append(
            {
                'epoch': epoch_index,
                'loss': round(final_loss, 6),
                'temperature': temperature_value,
                'projection_dim': projection_dim,
            }
        )

    payload = {
        'experiment_name': train_options['experiment_name'],
        'queue_size': int(train_options['queue_size']),
        'projection_dim': projection_dim,
        'epoch_count': epoch_count,
        'learning_rate': learning_rate,
        'temperature_value': temperature_value,
        'sample_count': len(feature_rows),
        'final_loss': round(final_loss, 6),
        'device_name': device_name,
        'status_text': '已完成',
        'metric_rows': metric_rows,
    }
    experiment_id = save_ybjy_feature_experiment(project_id, payload)
    return {
        'ok': True,
        'experiment_id': experiment_id,
        'message': '表征投影头基线训练已完成。',
    }


def build_ybjy_representation_lab_data(project_info, queue_size):
    queue_summary = build_ybjy_feature_queue_summary(project_info['id'], queue_size)
    experiment_rows = list_ybjy_feature_experiments_by_project(project_info['id'])
    for row in experiment_rows:
        try:
            row['metric_rows'] = json.loads(row.get('metric_json', '[]'))
        except json.JSONDecodeError:
            row['metric_rows'] = []
    return {
        'project_info': project_info,
        'queue_summary': queue_summary,
        'experiment_rows': experiment_rows,
        'queue_size': queue_size,
    }


def export_ybjy_representation_experiments_csv(project_info):
    experiment_rows = list_ybjy_feature_experiments_by_project(project_info['id'])
    file_buffer = io.StringIO()
    writer = csv.writer(file_buffer)
    writer.writerow(['项目名称', project_info['project_name']])
    writer.writerow(['导出内容', '冻结表征队列实验记录'])
    writer.writerow([])
    writer.writerow(
        [
            '实验编号',
            '实验名称',
            '队列长度',
            '投影维度',
            '训练轮次',
            '学习率',
            '温度系数',
            '样本数量',
            '最终损失',
            '运行设备',
            '状态',
            '创建时间',
        ]
    )
    for row in experiment_rows:
        writer.writerow(
            [
                row.get('id', ''),
                row.get('experiment_name', ''),
                row.get('queue_size', ''),
                row.get('projection_dim', ''),
                row.get('epoch_count', ''),
                row.get('learning_rate', ''),
                row.get('temperature_value', ''),
                row.get('sample_count', ''),
                row.get('final_loss', ''),
                row.get('device_name', ''),
                row.get('status_text', ''),
                row.get('created_at', ''),
            ]
        )
    export_name = f"{project_info['project_name']}_representation_experiments.csv"
    export_path = Path(EXPORT_DIR) / export_name
    with open(export_path, 'w', encoding='utf-8-sig', newline='') as export_file:
        export_file.write(file_buffer.getvalue())
    return export_name


def build_ybjy_research_fit_text():
    return (
        '本模块用于对已导入图片的冻结视觉语义特征进行队列化管理和轻量表征实验记录。'
        '系统不修改 CLIP 主干权重，只提供固定特征队列统计、'
        '标准投影头基线训练、损失记录和实验导出能力。'
        '该定位可以服务于冻结 CLIP 与动量对比学习相关研究方向的工程支撑。'
    )
