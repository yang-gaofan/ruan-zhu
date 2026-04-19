import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ybjy_config import ALLOWED_IMAGE_EXTENSIONS, APP_SHORT_NAME, APP_TITLE, EXPORT_DIR
from app.ybjy_database import list_ybjy_feedback_by_project, list_ybjy_samples_by_project


def is_allowed_ybjy_image(file_name):
    if '.' not in file_name:
        return False
    suffix = Path(file_name).suffix.lower()
    return suffix in ALLOWED_IMAGE_EXTENSIONS


def make_ybjy_safe_file_name(file_name):
    raw_name = Path(file_name).stem
    suffix = Path(file_name).suffix.lower()
    safe_name = ''.join(ch for ch in raw_name if ch.isalnum() or ch in ['_', '-'])
    if not safe_name:
        safe_name = 'sample_image'
    return safe_name + suffix


def build_ybjy_storage_name(original_name, sample_index):
    safe_name = make_ybjy_safe_file_name(original_name)
    return f'{sample_index:06d}_{safe_name}'


def calculate_ybjy_similarity_score(query_vector, target_vector):
    query_array = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    target_array = np.array(target_vector, dtype=np.float32).reshape(1, -1)
    score_value = cosine_similarity(query_array, target_array)[0][0]
    return float(score_value)


def build_ybjy_query_key(query_type, query_value):
    content_value = f'{query_type}::{query_value}'
    return hashlib.md5(content_value.encode('utf-8')).hexdigest()


def apply_ybjy_feedback_bonus(project_id, query_type, query_value, result_list):
    feedback_records = list_ybjy_feedback_by_project(project_id)
    query_key = build_ybjy_query_key(query_type, query_value)
    feedback_map = {}

    for item in feedback_records:
        current_key = build_ybjy_query_key(item['query_type'], item['query_value'])
        if current_key == query_key:
            feedback_map[item['target_sample_id']] = item['feedback_value']

    reranked_list = []
    for item in result_list:
        current_bonus = 0.0
        if item['id'] in feedback_map:
            current_bonus = 0.15 if feedback_map[item['id']] == 1 else -0.15
        item['feedback_bonus'] = current_bonus
        item['final_score'] = round(item['raw_score'] + current_bonus, 6)
        reranked_list.append(item)

    reranked_list.sort(key=lambda row: row['final_score'], reverse=True)
    return reranked_list


def run_ybjy_text_search(project_id, query_text, text_feature, top_k):
    sample_records = list_ybjy_samples_by_project(project_id)
    result_list = []

    for item in sample_records:
        target_feature = json.loads(item['feature_json'])
        raw_score = calculate_ybjy_similarity_score(text_feature, target_feature)
        result_list.append(
            {
                'id': item['id'],
                'file_name': item['file_name'],
                'relative_path': item['relative_path'],
                'raw_score': round(raw_score, 6),
            }
        )

    result_list.sort(key=lambda row: row['raw_score'], reverse=True)
    reranked_list = apply_ybjy_feedback_bonus(project_id, 'text', query_text.strip(), result_list)
    return reranked_list[:top_k]


def run_ybjy_image_search(project_id, source_sample_id, top_k):
    sample_records = list_ybjy_samples_by_project(project_id)
    source_record = None

    for item in sample_records:
        if item['id'] == source_sample_id:
            source_record = item
            break

    if source_record is None:
        return []

    source_feature = json.loads(source_record['feature_json'])
    result_list = []

    for item in sample_records:
        if item['id'] == source_sample_id:
            continue
        target_feature = json.loads(item['feature_json'])
        raw_score = calculate_ybjy_similarity_score(source_feature, target_feature)
        result_list.append(
            {
                'id': item['id'],
                'file_name': item['file_name'],
                'relative_path': item['relative_path'],
                'raw_score': round(raw_score, 6),
            }
        )

    result_list.sort(key=lambda row: row['raw_score'], reverse=True)
    reranked_list = apply_ybjy_feedback_bonus(project_id, 'image', str(source_sample_id), result_list)
    return reranked_list[:top_k]


def export_ybjy_project_csv(project_info, search_rows, export_type):
    file_buffer = io.StringIO()
    writer = csv.writer(file_buffer)
    writer.writerow(['软件全称', APP_TITLE])
    writer.writerow(['软件简称', APP_SHORT_NAME])
    writer.writerow(['项目名称', project_info['project_name']])
    writer.writerow(['导出类型', export_type])
    writer.writerow([])
    writer.writerow(['样本编号', '文件名称', '原始分数', '校正分值', '最终分数', '相对路径'])

    for row in search_rows:
        writer.writerow(
            [
                row.get('id', ''),
                row.get('file_name', ''),
                row.get('raw_score', ''),
                row.get('feedback_bonus', 0.0),
                row.get('final_score', row.get('raw_score', '')),
                row.get('relative_path', ''),
            ]
        )

    export_content = file_buffer.getvalue()
    export_name = f"{project_info['project_name']}_{export_type}_result.csv"
    export_path = Path(EXPORT_DIR) / export_name
    with open(export_path, 'w', encoding='utf-8-sig', newline='') as export_file:
        export_file.write(export_content)
    return export_name


def export_ybjy_feedback_csv(project_info):
    file_buffer = io.StringIO()
    writer = csv.writer(file_buffer)
    writer.writerow(['软件全称', APP_TITLE])
    writer.writerow(['软件简称', APP_SHORT_NAME])
    writer.writerow(['项目名称', project_info['project_name']])
    writer.writerow([])
    writer.writerow(['反馈编号', '样本编号', '文件名称', '查询类型', '查询内容', '反馈值', '记录时间'])

    feedback_rows = list_ybjy_feedback_by_project(project_info['id'])
    for row in feedback_rows:
        writer.writerow(
            [
                row.get('id', ''),
                row.get('target_sample_id', ''),
                row.get('file_name', ''),
                row.get('query_type', ''),
                row.get('query_value', ''),
                row.get('feedback_value', ''),
                row.get('created_at', ''),
            ]
        )

    export_content = file_buffer.getvalue()
    export_name = f"{project_info['project_name']}_feedback_record.csv"
    export_path = Path(EXPORT_DIR) / export_name
    with open(export_path, 'w', encoding='utf-8-sig', newline='') as export_file:
        export_file.write(export_content)
    return export_name


def build_ybjy_function_description_text():
    return (
        '轻量化图像样本语义管理与检索系统 V1.0 是一套面向本地图片样本整理、'
        '语义检索和结果归档场景的单机 Web 工具软件。系统以项目为基本管理单元，'
        '用户进入系统后可以创建不同的图片样本项目，填写项目名称和备注，并在项目内'
        '批量导入 JPG、PNG、BMP、WEBP 等常见格式的本地图片文件。系统在接收图片后，'
        '会将样本文件保存到本地项目目录，同时调用固定的图像语义特征提取组件生成图片'
        '特征向量，并把文件名称、存储路径、特征数据和导入时间统一写入 SQLite 数据库，'
        '形成可查询、可维护、可追踪的本地样本台账。'
        '\n\n'
        '系统提供文字检索图片和图片检索图片两类核心检索方式。文字检索图片功能允许用户'
        '输入中文或英文检索词，系统将检索文字转换为语义特征，并与当前项目内已导入图片'
        '的特征进行相似度计算，按照相关程度从高到低返回结果。图片检索图片功能允许用户'
        '从已有样本中选择一张图片作为查询来源，系统使用该图片的特征与同项目内其他样本'
        '逐项比对，输出相似图片列表。两类检索结果均展示原始相似度分数、人工校正分值和'
        '最终排序分数，方便用户理解结果排序依据。'
        '\n\n'
        '系统还提供人工校正与结果导出功能。用户可以对检索结果执行保留或降低操作，系统'
        '会记录查询类型、查询内容、目标样本、校正方向和记录时间。后续执行同一查询时，'
        '系统会读取历史校正记录，对对应样本进行轻量排序调整，形成检索、校正、再检索的'
        '业务闭环。导出中心支持将文字检索结果、图片检索结果和人工校正记录导出为 CSV 文件，'
        '便于后续整理、归档和材料编制。系统采用 Python、Flask、Jinja2、Bootstrap、SQLite '
        '和固定图像语义特征提取组件构建，可在普通 Windows 单机环境中部署运行，适用于中小'
        '规模图片样本管理、快速检索、人工复核和结果留存等应用场景。'
    )
