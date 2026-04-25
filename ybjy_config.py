from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

APP_TITLE = '基于冻结 CLIP 与动量对比学习的图像表征训练与检索系统 V1.0'
APP_SHORT_NAME = '图像表征训检系统'
SECRET_KEY = 'ybjy-system-secret-key-2026'

DATA_DIR = BASE_DIR / 'runtime_data'
DATABASE_DIR = DATA_DIR / 'database'
UPLOAD_DIR = DATA_DIR / 'uploads'
EXPORT_DIR = DATA_DIR / 'exports'
CACHE_DIR = DATA_DIR / 'cache'

DATABASE_PATH = DATABASE_DIR / 'yangben_yujian_system.db'
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_CONTENT_LENGTH = 30 * 1024 * 1024

CLIP_MODEL_NAME = 'ViT-B-32'
CLIP_PRETRAINED_NAME = 'laion2b_s34b_b79k'

TOP_K_DEFAULT = 12
BOOTSTRAP_SERVE_LOCAL = True
FUNCTION_DESCRIPTION_MIN_WORDS = 500
FUNCTION_DESCRIPTION_MAX_WORDS = 1300

for path_item in [DATA_DIR, DATABASE_DIR, UPLOAD_DIR, EXPORT_DIR, CACHE_DIR]:
    os.makedirs(path_item, exist_ok=True)
