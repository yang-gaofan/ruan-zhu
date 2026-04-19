import csv
import io
import json
from collections import Counter
from pathlib import Path

from ybjy_config import ALLOWED_IMAGE_EXTENSIONS, EXPORT_DIR
from app.ybjy_database import list_ybjy_feedback_by_project, list_ybjy_samples_by_project


def build_ybjy_extension_summary(sample_rows):
    counter = Counter()
    for row in sample_rows:
        suffix = Path(row.get('file_name', '')).suffix.lower()
        if suffix:
            counter[suffix] += 1
        else:
            counter['无扩展名'] += 1
    return [
        {
            'extension': extension,
            'count': count,
        }
        for extension, count in sorted(counter.items(), key=lambda item: item[0])
    ]


def build_ybjy_feedback_summary(feedback_rows):
    positive_count = 0
    negative_count = 0
    text_count = 0
    image_count = 0

    for row in feedback_rows:
        if row.get('feedback_value') == 1:
            positive_count += 1
        elif row.get('feedback_value') == -1:
            negative_count += 1

        if row.get('query_type') == 'text':
            text_count += 1
        elif row.get('query_type') == 'image':
            image_count += 1

    return {
        'total_count': len(feedback_rows),
        'positive_count': positive_count,
        'negative_count': negative_count,
        'text_count': text_count,
        'image_count': image_count,
    }


def build_ybjy_sample_feature_summary(sample_rows):
    vector_lengths = []
    invalid_count = 0

    for row in sample_rows:
        feature_json = row.get('feature_json') or ''
        try:
            feature_value = json.loads(feature_json)
        except json.JSONDecodeError:
            invalid_count += 1
            continue

        if isinstance(feature_value, list):
            vector_lengths.append(len(feature_value))
        else:
            invalid_count += 1

    if not vector_lengths:
        return {
            'feature_count': 0,
            'feature_dim': 0,
            'invalid_count': invalid_count,
        }

    most_common_dim = Counter(vector_lengths).most_common(1)[0][0]
    return {
        'feature_count': len(vector_lengths),
        'feature_dim': most_common_dim,
        'invalid_count': invalid_count,
    }


def build_ybjy_project_overview(project_info):
    sample_rows = list_ybjy_samples_by_project(project_info['id'])
    feedback_rows = list_ybjy_feedback_by_project(project_info['id'])
    extension_summary = build_ybjy_extension_summary(sample_rows)
    feedback_summary = build_ybjy_feedback_summary(feedback_rows)
    feature_summary = build_ybjy_sample_feature_summary(sample_rows)

    return {
        'project_info': project_info,
        'sample_count': len(sample_rows),
        'feedback_count': len(feedback_rows),
        'extension_summary': extension_summary,
        'feedback_summary': feedback_summary,
        'feature_summary': feature_summary,
        'allowed_extensions': sorted(ALLOWED_IMAGE_EXTENSIONS),
    }


def build_ybjy_operation_checklist(project_info):
    sample_rows = list_ybjy_samples_by_project(project_info['id'])
    feedback_rows = list_ybjy_feedback_by_project(project_info['id'])
    checklist_rows = []

    checklist_rows.append(
        {
            'name': '项目已创建',
            'status': '已完成' if project_info else '未完成',
            'note': '项目名称、备注和创建时间已写入数据库。',
        }
    )
    checklist_rows.append(
        {
            'name': '样本已导入',
            'status': '已完成' if sample_rows else '待完成',
            'note': f'当前项目已有 {len(sample_rows)} 张图片样本。',
        }
    )
    checklist_rows.append(
        {
            'name': '特征已保存',
            'status': '已完成' if all(row.get('feature_json') for row in sample_rows) and sample_rows else '待完成',
            'note': '每张图片导入后会保存本地特征向量。',
        }
    )
    checklist_rows.append(
        {
            'name': '检索可执行',
            'status': '已完成' if len(sample_rows) >= 1 else '待完成',
            'note': '文字检索至少需要一张样本，图片检索建议至少两张样本。',
        }
    )
    checklist_rows.append(
        {
            'name': '人工校正记录',
            'status': '已完成' if feedback_rows else '可选',
            'note': f'当前项目已有 {len(feedback_rows)} 条人工校正记录。',
        }
    )
    checklist_rows.append(
        {
            'name': '导出材料',
            'status': '可执行',
            'note': '导出中心可生成检索结果和人工校正记录 CSV。',
        }
    )

    return checklist_rows


def build_ybjy_project_report_text(project_info):
    overview = build_ybjy_project_overview(project_info)
    checklist_rows = build_ybjy_operation_checklist(project_info)
    feedback_summary = overview['feedback_summary']
    feature_summary = overview['feature_summary']

    lines = [
        f"项目名称：{project_info['project_name']}",
        f"项目备注：{project_info.get('project_note') or '暂无'}",
        f"创建时间：{project_info['created_at']}",
        '',
        f"图片样本数量：{overview['sample_count']}",
        f"人工校正记录：{overview['feedback_count']}",
        f"特征向量数量：{feature_summary['feature_count']}",
        f"主要特征维度：{feature_summary['feature_dim']}",
        f"异常特征记录：{feature_summary['invalid_count']}",
        '',
        '文件格式统计：',
    ]

    if overview['extension_summary']:
        for item in overview['extension_summary']:
            lines.append(f"- {item['extension']}：{item['count']} 个")
    else:
        lines.append('- 暂无样本文件')

    lines.extend(
        [
            '',
            '人工校正统计：',
            f"- 保留结果：{feedback_summary['positive_count']} 条",
            f"- 降低结果：{feedback_summary['negative_count']} 条",
            f"- 文字查询反馈：{feedback_summary['text_count']} 条",
            f"- 图片查询反馈：{feedback_summary['image_count']} 条",
            '',
            '项目核对清单：',
        ]
    )

    for row in checklist_rows:
        lines.append(f"- {row['name']}：{row['status']}，{row['note']}")

    return '\n'.join(lines)


def export_ybjy_project_inventory_csv(project_info):
    sample_rows = list_ybjy_samples_by_project(project_info['id'])
    file_buffer = io.StringIO()
    writer = csv.writer(file_buffer)

    writer.writerow(['项目名称', project_info['project_name']])
    writer.writerow(['项目备注', project_info.get('project_note', '')])
    writer.writerow(['创建时间', project_info['created_at']])
    writer.writerow([])
    writer.writerow(['样本编号', '文件名称', '相对路径', '导入时间', '特征长度'])

    for row in sample_rows:
        feature_length = 0
        try:
            feature_length = len(json.loads(row.get('feature_json', '[]')))
        except json.JSONDecodeError:
            feature_length = 0
        writer.writerow(
            [
                row.get('id', ''),
                row.get('file_name', ''),
                row.get('relative_path', ''),
                row.get('created_at', ''),
                feature_length,
            ]
        )

    export_name = f"{project_info['project_name']}_sample_inventory.csv"
    export_path = Path(EXPORT_DIR) / export_name
    with open(export_path, 'w', encoding='utf-8-sig', newline='') as export_file:
        export_file.write(file_buffer.getvalue())
    return export_name
