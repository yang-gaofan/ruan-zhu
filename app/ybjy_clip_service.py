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


YBJY_TEXT_ALIASES = {
    '人': ['person', 'human being'],
    '男人': ['man', 'male person'],
    '女人': ['woman', 'female person'],
    '戴眼镜的人': ['person wearing glasses', 'a person with glasses', 'a woman wearing glasses', 'a man wearing glasses'],
    '戴眼镜的女人': ['woman wearing glasses', 'female person with glasses'],
    '戴眼镜的男人': ['man wearing glasses', 'male person with glasses'],
    '狗': ['dog'],
    '小狗': ['dog', 'puppy'],
    '黄狗': ['yellow dog', 'golden retriever', 'Labrador retriever', 'light brown dog'],
    '黄色的狗': ['yellow dog', 'golden retriever', 'Labrador retriever', 'light brown dog'],
    '金色的狗': ['golden retriever', 'yellow dog', 'light brown dog'],
    '黑狗': ['black dog', 'black Labrador', 'dark dog'],
    '白狗': ['white dog', 'white puppy', 'samoyed'],
    '棕色的狗': ['brown dog', 'golden retriever', 'Labrador retriever'],
    '趴着的狗': ['dog lying down', 'lying dog', 'dog resting on the ground'],
    '躺着的狗': ['dog lying down', 'lying dog', 'dog resting'],
    '坐着的狗': ['sitting dog', 'dog sitting'],
    '奔跑的狗': ['running dog', 'dog running'],
    '猫': ['cat'],
    '小猫': ['cat', 'kitten'],
    '趴着的猫': ['cat lying down', 'lying cat', 'cat resting'],
    '鸟': ['bird'],
    '青蛙': ['frog'],
    '马': ['horse'],
    '飞机': ['airplane', 'airliner'],
    '汽车': ['car', 'automobile'],
    '跑车': ['sports car'],
    '卡车': ['truck', 'trailer truck'],
    '船': ['ship', 'boat'],
    '帆船': ['sailboat', 'schooner'],
}


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
        clean_text = text_value.strip()
        text_values = [clean_text]
        for key, alias_list in YBJY_TEXT_ALIASES.items():
            if key in clean_text:
                text_values.extend(alias_list)
        text_values = list(dict.fromkeys([value for value in text_values if value]))
        text_prompts = []
        for value in text_values:
            text_prompts.extend(
                [
                    value,
                    f'a photo of a {value}',
                    f'a clear photo of a {value}',
                    f'a close-up photo of a {value}',
                    f'a natural image of a {value}',
                ]
            )
        text_tokens = self.tokenizer(text_prompts).to(self.device)
        with torch.no_grad():
            feature_tensor = self.model.encode_text(text_tokens)
            feature_tensor = feature_tensor / feature_tensor.norm(dim=-1, keepdim=True)
            feature_tensor = feature_tensor.mean(dim=0)
        feature_array = feature_tensor.cpu().numpy()
        return self._normalize_feature(feature_array)


frozen_ybjy_clip_service = FrozenYbjyClipService()
