import numpy as np
import os
import torch
import certifi
from PIL import Image
from ybjy_config import CACHE_DIR, CLIP_MODEL_NAME, CLIP_PRETRAINED_NAME

HF_CACHE_DIR = CACHE_DIR / 'huggingface'
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault('HF_HOME', str(HF_CACHE_DIR))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TRANSFORMERS_CACHE', str(HF_CACHE_DIR / 'transformers'))
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())

import open_clip


class FrozenYbjyClipService:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.preprocess = None
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self._load_model_once()

    def _load_model_once(self):
        if self.model is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME,
                pretrained=CLIP_PRETRAINED_NAME,
                device=self.device
            )
            model.eval()
            self.model = model
            self.preprocess = preprocess

    def _normalize_feature(self, feature_array):
        feature_array = np.array(feature_array, dtype=np.float32)
        norm_value = np.linalg.norm(feature_array)
        if norm_value == 0:
            return feature_array.tolist()
        return (feature_array / norm_value).tolist()

    def encode_image_to_feature(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature_tensor = self.model.encode_image(image_tensor)
        feature_array = feature_tensor.squeeze(0).cpu().numpy()
        return self._normalize_feature(feature_array)

    def encode_text_to_feature(self, text_value):
        text_tokens = self.tokenizer([text_value]).to(self.device)
        with torch.no_grad():
            feature_tensor = self.model.encode_text(text_tokens)
        feature_array = feature_tensor.squeeze(0).cpu().numpy()
        return self._normalize_feature(feature_array)


frozen_ybjy_clip_service = FrozenYbjyClipService()
