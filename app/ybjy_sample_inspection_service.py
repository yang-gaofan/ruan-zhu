import csv
import io
import json
from pathlib import Path

from PIL import Image

from ybjy_config import EXPORT_DIR, UPLOAD_DIR
from app.ybjy_database import list_ybjy_samples_by_project


def format_ybjy_file_size(byte_count):
    if byte_count is None:
        return '未知'
    if byte_count < 1024:
        return f'{byte_count} B'
    if byte_count < 1024 * 1024:
        return f'{byte_count / 1024:.2f} KB'
    return f'{byte_count / 1024 / 1024:.2f} MB'


def calculate_ybjy_aspect_ratio(width, height):
    if not width or not height:
        return ''
    return f'{width}:{height}'


def read_ybjy_feature_length(feature_json):
    if not feature_json:
        return 0
    try:
        feature_value = json.loads(feature_json)
    except json.JSONDecodeError:
        return 0
    if not isinstance(feature_value, list):
        return 0
    return len(feature_value)


def inspect_ybjy_image_file(sample_row):
    relative_path = sample_row.get('relative_path', '')
    file_path = Path(UPLOAD_DIR) / relative_path
    file_exists = file_path.exists()
    file_size = file_path.stat().st_size if file_exists else None
    suffix = file_path.suffix.lower()

    image_width = 0
    image_height = 0
    image_mode = ''
    image_format = ''
    readable = False
    error_message = ''

    if file_exists:
        try:
            with Image.open(file_path) as image:
                image_width, image_height = image.size
                image_mode = image.mode
                image_format = image.format or suffix.replace('.', '').upper()
                readable = True
        except OSError as error:
            error_message = str(error)

    feature_length = read_ybjy_feature_length(sample_row.get('feature_json', ''))
    status_text = '正常'
    if not file_exists:
        status_text = '文件缺失'
    elif not readable:
        status_text = '图片不可读'
    elif feature_length <= 0:
        status_text = '特征异常'

    return {
        'id': sample_row.get('id', ''),
        'file_name': sample_row.get('file_name', ''),
        'relative_path': relative_path,
        'created_at': sample_row.get('created_at', ''),
        'file_exists': file_exists,
        'readable': readable,
        'file_size': file_size,
        'file_size_text': format_ybjy_file_size(file_size),
        'extension': suffix,
        'image_width': image_width,
        'image_height': image_height,
        'image_mode': image_mode,
        'image_format': image_format,
        'aspect_ratio': calculate_ybjy_aspect_ratio(image_width, image_height),
        'feature_length': feature_length,
        'status_text': status_text,
        'error_message': error_message,
    }


def build_ybjy_sample_inspection_rows(project_id):
    sample_rows = list_ybjy_samples_by_project(project_id)
    inspection_rows = []

    for sample_row in sample_rows:
        inspection_rows.append(inspect_ybjy_image_file(sample_row))

    return inspection_rows


def summarize_ybjy_inspection_rows(inspection_rows):
    total_count = len(inspection_rows)
    existing_count = 0
    readable_count = 0
    missing_count = 0
    abnormal_count = 0
    total_bytes = 0

    for row in inspection_rows:
        if row['file_exists']:
            existing_count += 1
        else:
            missing_count += 1

        if row['readable']:
            readable_count += 1

        if row['status_text'] != '正常':
            abnormal_count += 1

        if isinstance(row.get('file_size'), int):
            total_bytes += row['file_size']

    return {
        'total_count': total_count,
        'existing_count': existing_count,
        'readable_count': readable_count,
        'missing_count': missing_count,
        'abnormal_count': abnormal_count,
        'total_size_text': format_ybjy_file_size(total_bytes),
    }


def group_ybjy_inspection_by_status(inspection_rows):
    groups = {
        '正常': [],
        '文件缺失': [],
        '图片不可读': [],
        '特征异常': [],
    }

    for row in inspection_rows:
        status_text = row.get('status_text', '特征异常')
        if status_text not in groups:
            groups[status_text] = []
        groups[status_text].append(row)

    return groups


def build_ybjy_inspection_page_data(project_info):
    inspection_rows = build_ybjy_sample_inspection_rows(project_info['id'])
    summary = summarize_ybjy_inspection_rows(inspection_rows)
    status_groups = group_ybjy_inspection_by_status(inspection_rows)

    return {
        'project_info': project_info,
        'inspection_rows': inspection_rows,
        'summary': summary,
        'status_groups': status_groups,
    }


def export_ybjy_sample_inspection_csv(project_info):
    inspection_rows = build_ybjy_sample_inspection_rows(project_info['id'])
    file_buffer = io.StringIO()
    writer = csv.writer(file_buffer)

    writer.writerow(['项目名称', project_info['project_name']])
    writer.writerow(['项目备注', project_info.get('project_note', '')])
    writer.writerow(['创建时间', project_info['created_at']])
    writer.writerow([])
    writer.writerow(
        [
            '样本编号',
            '文件名称',
            '相对路径',
            '文件状态',
            '文件大小',
            '图片格式',
            '图片宽度',
            '图片高度',
            '颜色模式',
            '特征长度',
            '导入时间',
        ]
    )

    for row in inspection_rows:
        writer.writerow(
            [
                row['id'],
                row['file_name'],
                row['relative_path'],
                row['status_text'],
                row['file_size_text'],
                row['image_format'],
                row['image_width'],
                row['image_height'],
                row['image_mode'],
                row['feature_length'],
                row['created_at'],
            ]
        )

    export_name = f"{project_info['project_name']}_sample_inspection.csv"
    export_path = Path(EXPORT_DIR) / export_name
    with open(export_path, 'w', encoding='utf-8-sig', newline='') as export_file:
        export_file.write(file_buffer.getvalue())
    return export_name
