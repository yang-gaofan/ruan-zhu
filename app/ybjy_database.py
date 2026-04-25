import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from ybjy_config import DATABASE_PATH, UPLOAD_DIR


def _dict_factory(cursor, row):
    data = {}
    for index, column in enumerate(cursor.description):
        data[column[0]] = row[index]
    return data


@contextmanager
def open_yangben_yujian_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = _dict_factory
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_yangben_yujian_database():
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS ybjy_project (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                project_note TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
            '''
        )
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS ybjy_sample (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                feature_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES ybjy_project(id)
            )
            '''
        )
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS ybjy_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                target_sample_id INTEGER NOT NULL,
                query_type TEXT NOT NULL,
                query_value TEXT NOT NULL,
                feedback_value INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES ybjy_project(id),
                FOREIGN KEY(target_sample_id) REFERENCES ybjy_sample(id)
            )
            '''
        )
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS ybjy_feature_experiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                experiment_name TEXT NOT NULL,
                queue_size INTEGER NOT NULL,
                projection_dim INTEGER NOT NULL,
                epoch_count INTEGER NOT NULL,
                learning_rate REAL NOT NULL,
                temperature_value REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                final_loss REAL NOT NULL,
                device_name TEXT NOT NULL,
                status_text TEXT NOT NULL,
                metric_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES ybjy_project(id)
            )
            '''
        )


def create_ybjy_project(project_name, project_note):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            'INSERT INTO ybjy_project(project_name, project_note, created_at) VALUES (?, ?, ?)',
            (project_name.strip(), project_note.strip(), current_time)
        )
        return cursor.lastrowid


def list_ybjy_projects():
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            '''
            SELECT p.id, p.project_name, p.project_note, p.created_at,
                   COUNT(s.id) AS sample_count
            FROM ybjy_project p
            LEFT JOIN ybjy_sample s ON p.id = s.project_id
            GROUP BY p.id, p.project_name, p.project_note, p.created_at
            ORDER BY p.id DESC
            '''
        )
        return cursor.fetchall()


def get_ybjy_project(project_id):
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM ybjy_project WHERE id = ?', (project_id,))
        return cursor.fetchone()


def save_ybjy_sample(project_id, file_name, relative_path, feature_vector):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    feature_json = json.dumps(feature_vector)
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            '''
            INSERT INTO ybjy_sample(project_id, file_name, relative_path, feature_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (project_id, file_name, relative_path, feature_json, current_time)
        )
        return cursor.lastrowid


def list_ybjy_samples_by_project(project_id):
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            'SELECT * FROM ybjy_sample WHERE project_id = ? ORDER BY id DESC',
            (project_id,)
        )
        return cursor.fetchall()


def get_ybjy_sample(sample_id):
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM ybjy_sample WHERE id = ?', (sample_id,))
        return cursor.fetchone()


def delete_ybjy_sample(sample_id):
    sample_record = get_ybjy_sample(sample_id)
    if not sample_record:
        return False
    image_path = Path(UPLOAD_DIR) / sample_record['relative_path']
    if image_path.exists():
        os.remove(image_path)
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute('DELETE FROM ybjy_feedback WHERE target_sample_id = ?', (sample_id,))
        cursor.execute('DELETE FROM ybjy_sample WHERE id = ?', (sample_id,))
    return True


def delete_ybjy_project(project_id):
    project_record = get_ybjy_project(project_id)
    if not project_record:
        return False
    project_folder = Path(UPLOAD_DIR) / f'project_{project_id}'
    if project_folder.exists():
        shutil.rmtree(project_folder)
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute('DELETE FROM ybjy_feature_experiment WHERE project_id = ?', (project_id,))
        cursor.execute('DELETE FROM ybjy_feedback WHERE project_id = ?', (project_id,))
        cursor.execute('DELETE FROM ybjy_sample WHERE project_id = ?', (project_id,))
        cursor.execute('DELETE FROM ybjy_project WHERE id = ?', (project_id,))
    return True


def save_ybjy_feedback(project_id, target_sample_id, query_type, query_value, feedback_value):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            '''
            INSERT INTO ybjy_feedback(project_id, target_sample_id, query_type, query_value, feedback_value, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (project_id, target_sample_id, query_type, query_value, feedback_value, current_time)
        )
        return cursor.lastrowid


def list_ybjy_feedback_by_project(project_id):
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            '''
            SELECT f.*, s.file_name
            FROM ybjy_feedback f
            LEFT JOIN ybjy_sample s ON f.target_sample_id = s.id
            WHERE f.project_id = ?
            ORDER BY f.id DESC
            ''',
            (project_id,)
        )
        return cursor.fetchall()


def save_ybjy_feature_experiment(project_id, experiment_payload):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            '''
            INSERT INTO ybjy_feature_experiment(
                project_id,
                experiment_name,
                queue_size,
                projection_dim,
                epoch_count,
                learning_rate,
                temperature_value,
                sample_count,
                final_loss,
                device_name,
                status_text,
                metric_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                project_id,
                experiment_payload['experiment_name'],
                experiment_payload['queue_size'],
                experiment_payload['projection_dim'],
                experiment_payload['epoch_count'],
                experiment_payload['learning_rate'],
                experiment_payload['temperature_value'],
                experiment_payload['sample_count'],
                experiment_payload['final_loss'],
                experiment_payload['device_name'],
                experiment_payload['status_text'],
                json.dumps(experiment_payload['metric_rows'], ensure_ascii=False),
                current_time,
            )
        )
        return cursor.lastrowid


def list_ybjy_feature_experiments_by_project(project_id):
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            '''
            SELECT *
            FROM ybjy_feature_experiment
            WHERE project_id = ?
            ORDER BY id DESC
            ''',
            (project_id,)
        )
        return cursor.fetchall()


def get_ybjy_feature_experiment(experiment_id):
    with open_yangben_yujian_db() as connection:
        cursor = connection.cursor()
        cursor.execute(
            'SELECT * FROM ybjy_feature_experiment WHERE id = ?',
            (experiment_id,)
        )
        return cursor.fetchone()
